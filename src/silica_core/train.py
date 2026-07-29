import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from PIL import Image
from glob import glob

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datetime import datetime
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader, SubsetRandomSampler
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.segmentation import MeanIoU
import wandb
import diffusers
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from silica_core.utils.hypersim_dataset import get_hypersim_dataset_depth_normal
from silica_core.utils import metric
from silica_core.utils.metric import MetricTracker
from silica_core.utils.alignment import depth2disparity, disparity2depth, align_depth_least_square
from silica_core.model import SILICA
from silica_core.utils.image_utils import concatenate_images
from silica_core.utils.dataloader import Mirage18k, Trans10kDataset, Ref3DDataset, GDDDataset, GSDSDataset, GWDDataset

logger = get_logger(__name__, log_level="INFO")

def run_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, step, dataloader):
    
    eval_metrics = [
        "abs_relative_difference",
        "delta1_acc",
        "delta2_acc",
        "delta3_acc",
    ]

    min_depth = 1e-3
    max_depth = None

    device = accelerator.device
    
    # Initialize Model
    model = SILICA.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    if args.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()

    model = model.to(accelerator.device)
    model.set_progress_bar_config(disable=True)
    # using official torch metric without bg
    miou_metric = MeanIoU(
        num_classes=2, 
        include_background=False, 
        input_format="index"
    ).to(device)

    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]
    metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()

    for data in tqdm(dataloader, desc="validation"):
        x0 = data["pixel_values"].to(device, dtype=weight_dtype)
        gt_mask = data["seg_values"]
        depth_gt = data["depth_gt"].squeeze(0).numpy()

        prompts = [args.segmentation_task, args.depth_task]
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        # dual task emb
        vec_combined = torch.tensor([[1, 0], [0, 1]], device=device).float()
        task_emb_combined = torch.cat([torch.sin(vec_combined), torch.cos(vec_combined)], dim=-1)

        rgb_combined = x0.repeat(2, 1, 1, 1)

        with torch.no_grad(), accelerator.autocast():
            prompt_embeds = text_encoder(text_input_ids)[0]
            preds = model(
                x0=rgb_combined,
                prompt=None,                 
                prompt_embeds=prompt_embeds,
                task_emb=task_emb_combined,
                timesteps=[args.timestep],
                output_type="np",
                processing_res=0,
                match_input_res=True,
            ).images # (2, c, h, w)

        pred_seg = preds[0] #(1, c, h, w)
        pred_depth = preds[1]

        depth_pred = pred_depth.mean(axis=-1)
        valid_mask = (depth_gt > 1e-3)

        gt_disparity, gt_non_neg_mask = depth2disparity(
                depth=depth_gt, return_mask=True
            )
        
        pred_non_neg_mask = depth_pred > 0
        valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

        disparity_pred, scale, shift = align_depth_least_square(
            gt_arr=gt_disparity,
            pred_arr=depth_pred,
            valid_mask_arr=valid_nonnegative_mask,
            return_scale_shift=True
        )
        disparity_pred = np.clip(disparity_pred, a_min=1e-3, a_max=None)
        depth_pred = disparity2depth(disparity_pred)

        depth_pred = np.clip(depth_pred, a_min=min_depth, a_max=max_depth)
        depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

        depth_pred_ts = torch.from_numpy(depth_pred).to(device)
        depth_raw_ts = torch.from_numpy(depth_gt).to(device)
        valid_mask_ts = torch.from_numpy(valid_mask).to(device)

        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
            metric_tracker.update(_metric_name, _metric)
        
        # normalizing it to get evry faded mask
        p_min, p_max = pred_seg.min(), pred_seg.max()
        if p_max - p_min > 0.1:
            pred_seg = (pred_seg - p_min) / (p_max - p_min)

        # averaging rgb to gray
        pred_gray = np.mean(pred_seg, axis=-1)

        # now setting the threshold
        pred_mask = (pred_gray > 0.5).astype(np.uint8)

        gt_tensor = gt_mask
        gt_np = gt_tensor.numpy()
        gt_np = gt_np[0]
        
        # because the dataloader adds 2 more channels
        gt_np = np.mean(gt_np, axis=0)

        # as it is (-1, 1) making it (0, 1)
        if gt_np.max() > 1.1: gt_mask = (gt_np > 127).astype(np.uint8)
        elif gt_np.min() < 0: gt_mask = (gt_np > 0).astype(np.uint8)
        else: gt_mask = (gt_np > 0.5).astype(np.uint8)

        pred_metric_input = torch.from_numpy(pred_mask).long().unsqueeze(0).to(device)
        target_metric_input = torch.from_numpy(gt_mask).long().unsqueeze(0).to(device)

        miou_metric.update(pred_metric_input, target_metric_input)

    final_miou = miou_metric.compute()
    print(f"mIoU without background: {final_miou.item()}")

    final_metrics = metric_tracker.result()
    
    final_metrics["val_miou"] = final_miou.item()

    accelerator.log(final_metrics, step=step)
        
    del model
    del miou_metric
    torch.cuda.empty_cache()

    unet.train()

    return final_miou.item(), final_metrics["abs_relative_difference"]

def parse_args():
    parser = argparse.ArgumentParser(description="Silica training")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--non_ema_revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--wandb", type=str, default="wandb", help="Logging to wandb")
    default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument("--wandb_name", type=str, default=default_name, help="Logging name to wandb")
    parser.add_argument("--task_name", type=str, default="segmentation+depth")

    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--tracker_project_name", type=str, default="Silica")

    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--timestep", type=int, default=999, help="Fixed timestep")

    parser.add_argument("--segmentation_task", type=str, required=True)
    parser.add_argument("--depth_task", type=str, required=True)
    
    parser.add_argument("--train_data_dir_hypersim", type=str, default=None, help="Base dir for Hypersim")
    parser.add_argument("--train_data_dir_gsds", type=str, required=True, help="Base dir for GSDS Dataset")
    parser.add_argument("--train_data_dir_gdd", type=str, required=True, help="Base dir for GDD Dataset")
    parser.add_argument("--train_data_dir_trans10k", type=str, required=True, help="Base dir for Trans10k Dataset")
    parser.add_argument("--train_data_dir_ref3d", type=str, required=True, help="Base dir for Ref3D Dataset")
    parser.add_argument("--train_data_dir_m18k", type=str, required=True, help="Base dir for Mirage18k Dataset")
    parser.add_argument("--train_data_dir_gwd", type=str, required=True, help="Base dir for GWD Dataset")

    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)

    parser.add_argument("--validation_steps", type=int, default=2000, help="Run validation every X steps.")

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--truncnorm_min",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        choices=['instnorm','truncnorm','perscene_norm','disparity','trunc_disparity'],
        default='trunc_disparity',
        help="The normalization type of the depth. "
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--align_cam_normal",
        action="store_true",
    )

    parser.add_argument("--seed", type=int, default=42)   

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.wandb,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)

    text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
    
    vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision,
        class_embed_type="projection", projection_class_embeddings_input_dim=4, 
        low_cpu_mem_usage=False, device_map=None,
    )

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()
        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    #optimization thing to save memory
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_hypersim_dataset, preprocess_train_hypersim, collate_fn_hypersim = get_hypersim_dataset_depth_normal(
        args.train_data_dir_hypersim, 
        args.resolution, 
        args.random_flip, 
        norm_type=args.norm_type, 
        truncnorm_min=args.truncnorm_min, 
        align_cam_normal=args.align_cam_normal
    )
    
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            train_hypersim_dataset = train_hypersim_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset_hypersim = train_hypersim_dataset.with_transform(preprocess_train_hypersim)

    # Get the datasets and dataloaders
    train_dataset_gsds = GSDSDataset(base_dir=args.train_data_dir_gsds, split="train", resize=True, resolution=args.resolution)
    train_dataset_gdd = GDDDataset(base_dir=args.train_data_dir_gdd, split="train", resize=True, resolution=args.resolution)
    train_dataset_trans10k = Trans10kDataset(base_dir=args.train_data_dir_trans10k, split="train", resize=True, resolution=args.resolution)
    train_dataset_ref3d = Ref3DDataset(base_dir=args.train_data_dir_ref3d, split="train", resize=True, resolution=args.resolution)
    train_dataset_m18k = Mirage18k(base_dir=args.train_data_dir_m18k, split="train", resize=True, resolution=args.resolution)
    train_dataset_gwd = GWDDataset(base_dir=args.train_data_dir_gwd, split="train", resize=True, resolution=args.resolution)

    train_dataset = torch.utils.data.ConcatDataset([train_dataset_gsds, train_dataset_gdd, train_dataset_trans10k, train_dataset_gwd, train_dataset_ref3d, train_dataset_m18k])

    train_dataloader_hypersim = DataLoader(
        train_dataset_hypersim,
        shuffle=True,
        collate_fn=collate_fn_hypersim,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    train_dataloader_seg = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=4,
        drop_last=True
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_seg) / args.gradient_accumulation_steps)
    assert args.max_train_steps is not None or args.num_train_epochs is not None, "max_train_steps or num_train_epochs should be provided"
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, train_dataloader_seg, train_dataloader_hypersim, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader_seg, train_dataloader_hypersim, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_seg) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("task_name")
        accelerator.init_trackers(
            args.tracker_project_name,
            tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.wandb_name}"}
            }
        )

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"Num examples Hypersim = {len(train_dataset_hypersim)}")
    logger.info(f"Num examples Mirage18k = {len(train_dataset_m18k)}") 
    logger.info(f"Num examples Trans10k = {len(train_dataset_trans10k)}")
    logger.info(f"Num examples 3DRef = {len(train_dataset_ref3d)}") 
    logger.info(f'Num examples GDD = {len(train_dataset_gdd)}')
    logger.info(f'Num examples GSDS = {len(train_dataset_gsds)}')
    logger.info(f'Num examples GW Depth = {len(train_dataset_gwd)}')
    logger.info(f'Total Data Seg data = {len(train_dataset_gsds) + len(train_dataset_gdd) + len(train_dataset_ref3d) + len(train_dataset_trans10k) + len(train_dataset_m18k) + len(train_dataset_gwd)}')
 

    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Unet timestep = {args.timestep}")
    logger.info(f"  Task name: {args.task_name}")
    logger.info(f"  Output Workspace: {args.output_dir}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    best_combined_score = -float("inf")
    best_model_path = None

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    iter_hypersim = iter(train_dataloader_hypersim)

    val_dataset = Mirage18k(base_dir=args.train_data_dir_m18k, split="validation", resize=False, resolution=args.resolution)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4, pin_memory=True)

    for epoch in range(first_epoch, args.num_train_epochs):

        iter_seg = iter(train_dataloader_seg)
        
        train_loss = 0.0
        log_ann_loss = 0.0
        log_rgb_loss = 0.0

        for _ in range(len(train_dataloader_seg)):

            seg_batch = next(iter_seg)

            try: 
                hyper_batch = next(iter_hypersim)
            except StopIteration:
                iter_hypersim = iter(train_dataloader_hypersim)
                hyper_batch = next(iter_hypersim)

            with accelerator.accumulate(unet):
                rgb_latents = vae.encode(torch.cat((seg_batch["pixel_values"],hyper_batch["pixel_values"]), dim=0).to(weight_dtype)).latent_dist.sample()

                rgb_latents = rgb_latents * vae.config.scaling_factor

                target_latents = vae.encode(torch.cat((seg_batch["seg_values"],hyper_batch["depth_values"]), dim=0).to(weight_dtype)).latent_dist.sample()

                target_latents = target_latents * vae.config.scaling_factor

                bsz = target_latents.shape[0]
                bsz_per_task = int(bsz/2)

                # Set timestep
                timesteps = torch.tensor([args.timestep], device=target_latents.device).repeat(bsz)
                timesteps = timesteps.long()

                # Get the unet input
                unet_input = rgb_latents

                seg_prompt =  args.segmentation_task 
                depth_prompt = args.depth_task 

                prompts = [seg_prompt] * bsz_per_task + [depth_prompt] * bsz_per_task

                text_inputs = tokenizer(
                    prompts,
                    padding="max_length", 
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(target_latents.device)
                encoder_hidden_states = text_encoder(text_input_ids, return_dict=False)[0]

                # Get the target for loss
                target = target_latents

                # Get the task embedding
                task_emb_anno = torch.tensor([1, 0]).float().unsqueeze(0).to(accelerator.device)
                task_emb_anno = torch.cat([torch.sin(task_emb_anno), torch.cos(task_emb_anno)], dim=-1).repeat(bsz_per_task, 1)
                task_emb_rgb = torch.tensor([0, 1]).float().unsqueeze(0).to(accelerator.device)
                task_emb_rgb = torch.cat([torch.sin(task_emb_rgb), torch.cos(task_emb_rgb)], dim=-1).repeat(bsz_per_task, 1)
                task_emb = torch.cat((task_emb_anno, task_emb_rgb), dim=0)

                # Predict
                model_pred = unet(unet_input, timesteps, encoder_hidden_states, return_dict=False, class_labels=task_emb)[0]

                # Compute loss
                anno_loss = F.mse_loss(model_pred[:bsz_per_task].float(), target[:bsz_per_task].float(), reduction="mean")
                rgb_loss = F.mse_loss(model_pred[bsz_per_task:].float(), target[bsz_per_task:].float(), reduction="mean")
                loss = anno_loss + rgb_loss

                # Gather loss
                avg_anno_loss = accelerator.gather(anno_loss.repeat(args.train_batch_size)).mean()
                log_ann_loss += avg_anno_loss.item() / args.gradient_accumulation_steps
                avg_rgb_loss = accelerator.gather(rgb_loss.repeat(args.train_batch_size)).mean()
                log_rgb_loss += avg_rgb_loss.item() / args.gradient_accumulation_steps
                train_loss = log_ann_loss + log_rgb_loss

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"train_loss": loss.detach().item(), 
                    "segmentation_loss": anno_loss.detach().item(), 
                    "depth_loss": rgb_loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss,
                                "segmentation_loss": log_ann_loss,
                                "depth_loss": log_rgb_loss},
                                 step=global_step)
                train_loss = 0.0
                log_ann_loss = 0.0
                log_rgb_loss = 0.0

                checkpointing_steps = args.checkpointing_steps
                validation_steps = args.validation_steps

                if accelerator.is_main_process:
                    if global_step % checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % validation_steps == 0:
                        if accelerator.is_main_process:
                            current_miou, current_abs_rel = run_validation(
                                vae,
                                text_encoder,
                                tokenizer,
                                unet,
                                args,
                                accelerator,
                                weight_dtype,
                                global_step,
                                val_dataloader 
                            )
                            
                            current_combined_score = (current_miou) - current_abs_rel
                            
                            logger.info(f"Validation at step {global_step}: mIoU={current_miou:.4f}, AbsRel={current_abs_rel:.4f} | Combined Score={current_combined_score:.4f}")

                            if current_combined_score > best_combined_score:
                                best_combined_score = current_combined_score
                                
                                unwrapped_unet = unwrap_model(unet)
                                best_model_to_save = SILICA.from_pretrained(
                                    args.pretrained_model_name_or_path,
                                    text_encoder=text_encoder,
                                    vae=vae,
                                    unet=unwrapped_unet,
                                    revision=args.revision,
                                    variant=args.variant,
                                )
                                
                                best_save_dir = os.path.join(args.output_dir, f"gs-{global_step}_abs-{current_abs_rel:.2f}_iou-{current_miou:.2f}")
                                
                                best_model_to_save.save_pretrained(best_save_dir)
                                logger.info(f"*** New Best Model Saved! Score: {best_combined_score:.4f} at step {global_step} ***")

                                if best_model_path is not None and os.path.exists(best_model_path):
                                    shutil.rmtree(best_model_path)
                                    logger.info(f"Cleaned up previous best model: {best_model_path}")
                                
                                best_model_path = best_save_dir

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    accelerator.end_training()

if __name__ == "__main__":
    main()