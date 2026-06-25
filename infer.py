import argparse
import os
import time
from pathlib import Path
from contextlib import nullcontext

import torch
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

from model import SILICA
from utils.image_utils import seed_all, colorize_depth_map

depth_scale = 1.0

def load_intrinsics(file_path):
    with open(file_path, 'r') as f:
        values = f.read().strip().split()
        
        return {
            'fx': float(values[0]),
            'fy': float(values[1]),
            'cx': float(values[2]),
            'cy': float(values[3]),
        }

# backprojection
def depth_to_pointcloud(depth_image, intrinsics, scale=1.0, color_image=None):
    h, w = depth_image.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    u = u.flatten()
    v = v.flatten()
    z = depth_image.flatten()

    # remove invalid depth
    valid_mask = z > 0
    u = u[valid_mask]
    v = v[valid_mask]
    z = z[valid_mask]
    
    z_meters = z * scale

    # perspective back projection
    x_meters = (u - intrinsics['cx']) * z_meters / intrinsics['fx']
    y_meters = (v - intrinsics['cy']) * z_meters / intrinsics['fy']

    points_3d = np.stack((x_meters, y_meters, z_meters), axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    if color_image is not None:
        c = color_image.reshape(-1, 3)
        c = c[valid_mask]
        pcd.colors = o3d.utility.Vector3dVector(c)

    return pcd

def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity

def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)

def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=False,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # (h, w)
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    
    start = time.perf_counter()
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    elapsed = time.perf_counter() - start

    scale, shift = X
    aligned_pred = pred_arr * scale + shift

    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred, elapsed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output_vis")
    parser.add_argument("--processing_res", type=int, default=768)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--input_gt_dir", type=str, default=None)
    parser.add_argument("--input_gt_file", type=str, default=None)
    parser.add_argument("--intrinsics_file", type=str, default=None, help="Path to a txt file containing fx fy cx cy separated by space")
    
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    # output dirs
    output_dir_depth_color = os.path.join(args.output_dir, 'depth_vis')
    output_dir_depth_npy = os.path.join(args.output_dir, 'depth_np')
    output_dir_seg = os.path.join(args.output_dir, 'segmentation')
    overlay_path = os.path.join(args.output_dir, "overlay")
    
    output_mixed_depth_np = os.path.join(args.output_dir, 'pred_depth_aligned_np')
    output_depth_pcd = os.path.join(args.output_dir, 'pred_pcd')

    for d in [output_dir_depth_color, output_dir_depth_npy, output_dir_seg, overlay_path, output_mixed_depth_np, output_depth_pcd]:
        os.makedirs(d, exist_ok=True)

    input_dir = Path(args.input_dir)
    images = []
    gt_images = []

    if args.input_file:
        with open(args.input_file, 'r') as f:
            relative_paths = [line.strip() for line in f if line.strip()]
            images = [input_dir / rel_path for rel_path in relative_paths]
            
            if args.input_gt_dir:
                if args.input_gt_file:
                    with open(args.input_gt_file, 'r') as gf:
                        gt_rel_paths = [line.strip() for line in gf if line.strip()]
                    gt_images = [Path(args.input_gt_dir) / rel_path for rel_path in gt_rel_paths]
                else:
                    gt_images = [Path(args.input_gt_dir) / Path(rel_path).with_suffix('.png') for rel_path in relative_paths]
    else:
        print(f"==> Scanning directory: {args.input_dir}")
        images = sorted(list(input_dir.rglob("*.png")) + 
                        list(input_dir.rglob("*.jpg")) + 
                        list(input_dir.rglob("*.JPEG")) + 
                        list(input_dir.rglob("*.jpeg")))
        
        if args.input_gt_dir:
            for img_path in images:
                rel_path = img_path.relative_to(input_dir)
                gt_images.append(Path(args.input_gt_dir) / rel_path.with_suffix('.png'))

    print(f"==> Found {len(images)} RGB images.")
    if args.input_gt_dir:
        print(f"==> GT alignment enabled. Expecting {len(gt_images)} GT depth maps.")
    
    device = "cuda" 
    dtype = torch.float16 
    
    print("Loading model...")
    model = SILICA.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=dtype).to(device)

    if args.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()

    model.set_progress_bar_config(disable=True)

    iterable = zip(images, gt_images) if args.input_gt_dir else zip(images, [None]*len(images))

    for img_path, gt_path in tqdm(iterable, total=len(images)):
        pil = Image.open(img_path).convert("RGB")
        rgb_np = np.array(pil)

        rgb = torch.tensor(rgb_np).permute(2,0,1).unsqueeze(0).float()
        rgb = rgb/127.5 - 1
        rgb = rgb.to(device)

        vec_combined = torch.tensor([[1, 0], [0, 1]], device=device).float()
        task_emb_combined = torch.cat([torch.sin(vec_combined), torch.cos(vec_combined)], dim=-1)
        rgb_combined = rgb.repeat(2, 1, 1, 1)

        with torch.no_grad():
            autocast_ctx = torch.autocast("cuda") if device=="cuda" else nullcontext()
            
            with autocast_ctx:
                preds = model(
                    x0=rgb_combined,
                    prompt=["predict glass segmentation", "predict depth map"],
                    task_emb=task_emb_combined,
                    timesteps=[args.timestep],
                    output_type="np",
                    processing_res=args.processing_res,
                    match_input_res=True,
                ).images # (2, h, w, c)

        pred_seg = preds[0]
        pred_depth = preds[1]

        depth_npy = pred_depth.mean(axis=-1) 
        depth_color = colorize_depth_map(depth_npy)
        depth_color.save(os.path.join(output_dir_depth_color, f'{img_path.name}'))
        np.save(os.path.join(output_dir_depth_npy, f'{img_path.stem}.npy'), depth_npy)

        p_min, p_max = pred_seg.min(), pred_seg.max()
        if p_max - p_min > 0.1:
            pred_seg = (pred_seg - p_min) / (p_max - p_min)

        pred_gray = np.max(pred_seg, axis=-1)
        pred_mask = (pred_gray > args.threshold).astype(np.uint8) * 255

        Image.fromarray(pred_mask).save(Path(output_dir_seg) / img_path.name)

        overlay = rgb_np.copy()
        color = np.array([255, 0, 0], dtype=np.uint8)
        alpha = 0.4
        mask_bool = pred_mask.astype(bool)
        overlay[mask_bool] = ((1 - alpha) * overlay[mask_bool] + alpha * color).astype(np.uint8)
        Image.fromarray(overlay).save(os.path.join(overlay_path, f"{img_path.stem}.png"))

        if args.input_gt_dir and gt_path and os.path.exists(gt_path):
            gt_np = np.array(Image.open(gt_path)).astype(np.float32) / 1000.0
            
            background_mask = pred_mask == 0
            glass_mask = pred_mask > 0
            
            mixed_depth = gt_np.copy().astype(np.float32)
            valid_mask = (gt_np > 1e-3) & (depth_npy > 1e-3) & background_mask

            gt_disparity, gt_non_neg_mask = depth2disparity(depth=gt_np, return_mask=True)
            pred_non_neg_mask = depth_npy > 0
            valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

            disparity_pred, t = align_depth_least_square(
                gt_arr=gt_disparity,
                pred_arr=depth_npy,
                valid_mask_arr=valid_nonnegative_mask,
                return_scale_shift=False
            )
            
            disparity_pred = np.clip(disparity_pred, a_min=1e-3, a_max=None)
            pred_depth_aligned = disparity2depth(disparity_pred)

            if pred_depth_aligned.ndim == 3:
                pred_depth_aligned = pred_depth_aligned.squeeze()

            mixed_depth[glass_mask] = pred_depth_aligned[glass_mask]
            np.save(os.path.join(output_mixed_depth_np, f'{img_path.stem}_mixed.npy'), mixed_depth)

            color_map = rgb_np.astype(np.float32) / 255.0

            intrinsics = load_intrinsics(args.intrinsics_file)

            pcd_injected = depth_to_pointcloud(
                mixed_depth, 
                intrinsics=intrinsics, 
                scale=depth_scale, 
                color_image=color_map
            )
            file_path_pcd = os.path.join(output_depth_pcd, f'{img_path.stem}.pcd')
            o3d.io.write_point_cloud(file_path_pcd, pcd_injected, write_ascii=False, compressed=True)

if __name__=="__main__":
    main()