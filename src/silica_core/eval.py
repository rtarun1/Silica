import argparse
import logging
import os
import sys
from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image
from tabulate import tabulate
from torch.utils.data import DataLoader
from torchmetrics.segmentation import MeanIoU
from tqdm.auto import tqdm

from silica_core.utils import metric
from silica_core.utils.metric import MetricTracker
from silica_core.utils.alignment import depth2disparity, disparity2depth, align_depth_least_square
from silica_core.utils.image_utils import colorize_depth_map
from silica_core.utils.dataloader import Mirage18k
from silica_core.model import SILICA

eval_metrics = [
    "abs_relative_difference",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc"
]

min_depth = 1e-3
max_depth = None

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Depth and Segmentation mIoU on Mirage18k")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_base_dir", type=str, required=True, help="Base directory for the Mirage18k")
    parser.add_argument("--pred_dir", type=str, default=None, help="Directory containing pre-computed depth .npy files")
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory containing pre-computed segmentation mask images")
    parser.add_argument("--alignment", type=str, default="least_square_disparity")
    parser.add_argument("--output_dir", type=str, default="val_outputs")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--processing_res", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--generate", action="store_true", help="Generate predictions using the model")
    parser.add_argument("--save_pred_vis", action="store_true", help="Save visual outputs and overlays")
    parser.add_argument("--segmentation_task", type=str, default="predict glass segmentation")
    parser.add_argument("--depth_task", type=str, default="predict depth")
    
    return parser.parse_args()

def align_prediction(depth_pred, target_depth, valid_mask, align_method):
    """Helper function to align depth based on the requested method and target."""
    if align_method == "least_square":
        aligned_pred, scale, shift = align_depth_least_square(
            gt_arr=target_depth,
            pred_arr=depth_pred,
            valid_mask_arr=valid_mask,
            return_scale_shift=True
        )   
    elif align_method == "least_square_disparity":
        gt_disparity, gt_non_neg_mask = depth2disparity(
            depth=target_depth, return_mask=True
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
        aligned_pred = disparity2depth(disparity_pred)
    elif align_method == "metric":
        aligned_pred = depth_pred.copy()
    else:
        aligned_pred = depth_pred.copy()

    aligned_pred = np.clip(aligned_pred, a_min=min_depth, a_max=max_depth)
    aligned_pred = np.clip(aligned_pred, a_min=1e-6, a_max=None)
    return aligned_pred

def evaluation_silica():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    model = None
    if args.generate:
        if not args.pretrained_model_name_or_path:
            raise ValueError("--pretrained_model_name_or_path must be provided if --generate is used.")
            
        model = SILICA.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=dtype
        ).to(device)

        if args.enable_xformers_memory_efficient_attention:
            model.enable_xformers_memory_efficient_attention()

        model.set_progress_bar_config(disable=True)
    else:
        if (not args.pred_dir or not os.path.exists(args.pred_dir)) and (not args.mask_dir or not os.path.exists(args.mask_dir)):
            raise ValueError("Valid '--pred_dir' and/or '--mask_dir' must be provided when '--generate' is not set.")

    depth_dataset = Mirage18k(base_dir=args.dataset_base_dir, split="depth", resize=False)
    depth_loader = DataLoader(depth_dataset, batch_size=1, num_workers=4, pin_memory=True)

    metric_funcs = [getattr(metric, _met) for _met in eval_metrics]
    depth_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
    depth_tracker.reset()

    csv_path_depth = os.path.join(args.output_dir, "per_sample_depth_evaluation.csv")
    with open(csv_path_depth, "w+") as f:
        f.write("filename," + ",".join([m.__name__ for m in metric_funcs]) + "\n")

    if args.save_pred_vis:
        save_np_path = os.path.join(args.output_dir, "np")
        save_vis_path = os.path.join(args.output_dir, "vis")
        os.makedirs(save_np_path, exist_ok=True)
        os.makedirs(save_vis_path, exist_ok=True)

    logging.info(f"Starting Depth Evaluation on Mirage18k (split='depth', {len(depth_dataset)} samples)...")
    
    for data in tqdm(depth_loader, desc="Evaluating Depth"):
        raw_depth = data["depth_raw"].squeeze(0).numpy()
        depth_gt = data["depth_gt"].squeeze(0).numpy()
        pred_name = data["filename"][0]
        base_name = os.path.splitext(pred_name)[0]

        if args.generate:
            pixel_values = data["pixel_values"].to(device, dtype=dtype)
            vec_combined = torch.tensor([[1, 0], [0, 1]], device=device).float()
            task_emb_combined = torch.cat([torch.sin(vec_combined), torch.cos(vec_combined)], dim=-1)
            rgb_combined = pixel_values.repeat(2, 1, 1, 1)
            
            with torch.no_grad():
                autocast_ctx = torch.autocast(device.type) if torch.cuda.is_available() else nullcontext()
                with autocast_ctx:
                    preds = model(
                        x0=rgb_combined,
                        prompt=[args.segmentation_task, args.depth_task],
                        task_emb=task_emb_combined,
                        timesteps=[999],
                        output_type="np",
                        processing_res=args.processing_res,
                        match_input_res=True 
                    ).images

            depth_pred_base = preds[1].mean(axis=-1)
        else:
            pred_path = os.path.join(args.pred_dir, f"{base_name}.npy")
            if not os.path.exists(pred_path):
                continue
            depth_pred_base = np.load(pred_path)

        if depth_pred_base.ndim > 2:
            depth_pred_base = depth_pred_base.squeeze()

        valid_mask_all = (depth_gt > 1e-3) & (raw_depth > 1e-3)
        eval_mask_all  = (depth_gt > 1e-3)

        depth_pred_aligned = align_prediction(
            depth_pred_base, 
            target_depth=depth_gt, 
            valid_mask=valid_mask_all, 
            align_method=args.alignment
        )

        depth_gt_ts = torch.from_numpy(depth_gt).to(device)
        pred_ts = torch.from_numpy(depth_pred_aligned).to(device)
        mask_ts = torch.from_numpy(eval_mask_all).to(device)

        sample_depth_metrics = []
        for met_func in metric_funcs:
            _metric_name = met_func.__name__
            _metric_val = met_func(pred_ts, depth_gt_ts, mask_ts).item()
            sample_depth_metrics.append(str(_metric_val))
            depth_tracker.update(_metric_name, _metric_val)

        with open(csv_path_depth, "a+") as f:
            f.write(f"{pred_name}," + ",".join(sample_depth_metrics) + "\n")

        if args.save_pred_vis:
            depth_pred_vis = colorize_depth_map(pred_ts.cpu(), reverse_color=True)
            depth_pred_vis.save(os.path.join(save_vis_path, f"{base_name}.png"))
            np.save(os.path.join(save_np_path, f"{base_name}.npy"), pred_ts.cpu().numpy())


    seg_dataset = Mirage18k(base_dir=args.dataset_base_dir, split="test", resize=False)
    seg_loader = DataLoader(seg_dataset, batch_size=1, num_workers=4, pin_memory=True)

    miou_metric = MeanIoU(
        num_classes=2,
        include_background=False,
        input_format="index"
    ).to(device)

    csv_path_seg = os.path.join(args.output_dir, "per_sample_seg_evaluation.csv")
    with open(csv_path_seg, "w+") as f:
        f.write("filename,mIoU\n")

    if args.save_pred_vis:
        save_seg_path = os.path.join(args.output_dir, "seg_overlay")
        os.makedirs(save_seg_path, exist_ok=True)

    logging.info(f"\nStarting Segmentation Evaluation on Mirage18k (split='test', {len(seg_dataset)} samples)...")
    
    for data in tqdm(seg_loader, desc="Evaluating Segmentation"):
        pred_name = data["filename"][0]
        base_name = os.path.splitext(pred_name)[0]

        # Extract and format Ground Truth Mask
        gt_np = data["seg_values"].squeeze(0).cpu().numpy()
        if gt_np.ndim == 3:
            gt_np = np.mean(gt_np, axis=0)
        
        if gt_np.max() > 1.1: gt_mask = (gt_np > 127).astype(np.uint8)
        elif gt_np.min() < 0: gt_mask = (gt_np > 0).astype(np.uint8)
        else: gt_mask = (gt_np > 0.5).astype(np.uint8)

        pred_mask = None

        if args.generate:
            pixel_values = data["pixel_values"].to(device, dtype=dtype)
            vec_combined = torch.tensor([[1, 0], [0, 1]], device=device).float()
            task_emb_combined = torch.cat([torch.sin(vec_combined), torch.cos(vec_combined)], dim=-1)
            rgb_combined = pixel_values.repeat(2, 1, 1, 1)

            with torch.no_grad():
                autocast_ctx = torch.autocast(device.type) if torch.cuda.is_available() else nullcontext()
                with autocast_ctx:
                    preds = model(
                        x0=rgb_combined,
                        prompt=[args.segmentation_task, args.depth_task],
                        task_emb=task_emb_combined,
                        timesteps=[999],
                        output_type="np",
                        processing_res=args.processing_res,
                        match_input_res=True 
                    ).images
            
            pred_seg = preds[0].astype(np.float32)
            p_min, p_max = pred_seg.min(), pred_seg.max()
            if p_max - p_min > 0.1:
                pred_seg = (pred_seg - p_min) / (p_max - p_min)
            pred_gray = np.mean(pred_seg, axis=-1)
            pred_mask = (pred_gray > args.threshold).astype(np.uint8)
            
        else:
            if args.mask_dir:
                mask_path = os.path.join(args.mask_dir, f"{base_name}.png")
                if os.path.exists(mask_path):
                    ext_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if ext_mask is not None:
                        pred_mask = (ext_mask > (args.threshold * 255 if ext_mask.max() > 1 else args.threshold)).astype(np.uint8)

        if pred_mask is not None:
            pred_metric_input = torch.from_numpy(pred_mask).long().unsqueeze(0).to(device)
            target_metric_input = torch.from_numpy(gt_mask).long().unsqueeze(0).to(device)
            
            sample_miou_tracker = MeanIoU(num_classes=2, include_background=False, input_format="index").to(device)
            sample_miou_tracker.update(pred_metric_input, target_metric_input)
            sample_miou = sample_miou_tracker.compute().item()

            miou_metric.update(pred_metric_input, target_metric_input)

            with open(csv_path_seg, "a+") as f:
                f.write(f"{pred_name},{sample_miou:.5f}\n")

            if args.save_pred_vis and "pixel_values" in data:
                rgb = data["pixel_values"].squeeze(0).detach().cpu().float()
                rgb = (rgb + 1) / 2
                rgb = rgb.permute(1, 2, 0).numpy()
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

                overlay = rgb.copy()
                color = np.array([255, 0, 0], dtype=np.uint8)
                mask_bool = pred_mask.astype(bool)
                overlay[mask_bool] = ((1 - 0.4) * overlay[mask_bool] + 0.4 * color).astype(np.uint8)

                Image.fromarray(overlay).save(os.path.join(save_seg_path, f"{base_name}.png"))

    final_miou = miou_metric.compute().item()

    eval_text = "Mirage18k Final Evaluation Summary:\n"
    eval_text += f"Alignment (Depth): {args.alignment}\n"
    eval_text += f"Segmentation mIoU (split='test'): {final_miou:.5f}\n"
    eval_text += "="*60 + "\n\n"

    eval_text += "--- Depth Metrics (split='depth', all_align_all_eval) ---\n"
    eval_text += tabulate([depth_tracker.result().keys(), depth_tracker.result().values()]) + "\n\n"

    metrics_filename = f"eval_metrics-{args.alignment}.txt" if args.alignment else "eval_metrics.txt"
    _save_to = os.path.join(args.output_dir, metrics_filename)
    
    with open(_save_to, "w+") as f:
        f.write(eval_text)

    logging.info(f"Evaluation metrics saved to {_save_to}")
    print(eval_text)

    return depth_tracker, final_miou

if __name__ == "__main__":
    evaluation_silica()