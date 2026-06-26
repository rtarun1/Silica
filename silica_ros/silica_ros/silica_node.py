#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge
import message_filters

import torch
import numpy as np
import cv2
import time
import threading
from contextlib import nullcontext

# ==========================================
# IMPORT FROM YOUR NEW SILICA PYTHON PACKAGE
# ==========================================
from silica_core.model import SILICA
from silica_core.infer import align_depth_least_square, depth2disparity, disparity2depth
from silica_core.utils.image_utils import seed_all, colorize_depth_map



class SilicaNode(Node):
    def __init__(self):
        super().__init__('silica_node')
        
        # --- Parameters ---
        # Note: Model path defaults to HuggingFace hub repo name or local dir
        self.declare_parameter('pretrained_model_name_or_path', 'silica-mirage/silica-v1-0')
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('processing_res', 512)
        self.declare_parameter('timestep', 999)
        self.declare_parameter('enable_xformers', True)
        self.declare_parameter('seed', 42)
        self.declare_parameter('pc_step', 4)  
        
        self.declare_parameter('rgb_topic_in', '/camera1/camera1/color/image_raw')
        self.declare_parameter('depth_topic_in', '/camera1/camera1/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera1/camera1/color/camera_info')
        
        self.declare_parameter('rgb_topic_out', 'image_glass')
        self.declare_parameter('depth_topic_out', 'depth_glass')
        self.declare_parameter('pointcloud_topic_out', 'points_glass')
        
        model_path = self.get_parameter('pretrained_model_name_or_path').value
        self.threshold = self.get_parameter('threshold').value
        self.processing_res = self.get_parameter('processing_res').value
        self.timestep = self.get_parameter('timestep').value
        enable_xformers = self.get_parameter('enable_xformers').value
        seed = self.get_parameter('seed').value
        self.pc_step = self.get_parameter('pc_step').value
        
        self.cam_matrix = None
        self.dist_coeffs = None
        
        self.frame_lock = threading.Lock()
        self.latest_frames = None

        seed_all(seed)
        
        # --- Load Model ---
        self.get_logger().info(f"Loading SILICA from {model_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16
        
        self.model = SILICA.from_pretrained(model_path, torch_dtype=dtype).to(self.device)
        
        if enable_xformers:
            self.model.enable_xformers_memory_efficient_attention()
            
        self.model.set_progress_bar_config(disable=True)
        self.get_logger().info("Model loaded successfully.")
        
        # --- ROS Setup ---
        self.bridge = CvBridge()
        
        self.pub_rgb = self.create_publisher(Image, self.get_parameter('rgb_topic_out').value, 10)
        self.pub_depth = self.create_publisher(Image, self.get_parameter('depth_topic_out').value, 10)
        self.pub_pc = self.create_publisher(PointCloud2, self.get_parameter('pointcloud_topic_out').value, 10)
        
        self.sub_info = self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self.info_callback, 10)
        
        self.sub_rgb = message_filters.Subscriber(self, Image, self.get_parameter('rgb_topic_in').value)
        self.sub_depth = message_filters.Subscriber(self, Image, self.get_parameter('depth_topic_in').value)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], queue_size=2, slop=0.1)
        self.ts.registerCallback(self.sync_callback)

        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()

        self.get_logger().info("Node initialized. Waiting for CameraInfo and frames...")

    def info_callback(self, msg):
        if self.cam_matrix is None:
            self.cam_matrix = np.array(msg.k).reshape((3, 3)).astype(np.float32)
            self.dist_coeffs = np.array(msg.d, dtype=np.float32)
            self.get_logger().info("Camera parameters received.")

    def sync_callback(self, msg_rgb, msg_depth):
        with self.frame_lock:
            self.latest_frames = (msg_rgb, msg_depth)

    def inference_loop(self):
        while rclpy.ok():
            frames = None
            with self.frame_lock:
                if self.latest_frames is not None:
                    frames = self.latest_frames
                    self.latest_frames = None 
            
            if frames is None:
                time.sleep(0.005)
                continue
                
            if self.cam_matrix is None or self.dist_coeffs is None:
                continue

            msg_rgb, msg_depth = frames

            try:
                rgb_np_raw = self.bridge.imgmsg_to_cv2(msg_rgb, desired_encoding="rgb8")
                depth_np_raw = self.bridge.imgmsg_to_cv2(msg_depth, desired_encoding="16UC1")
                
                gt_np = depth_np_raw.astype(np.float32) / 1000.0
                rgb_np = cv2.undistort(rgb_np_raw, self.cam_matrix, self.dist_coeffs)

                rgb_tensor = torch.tensor(rgb_np).permute(2, 0, 1).unsqueeze(0).float()
                rgb_tensor = rgb_tensor / 127.5 - 1
                rgb_tensor = rgb_tensor.to(self.device)

                vec_combined = torch.tensor([[1, 0], [0, 1]], device=self.device, dtype=torch.float16)
                task_emb_combined = torch.cat([torch.sin(vec_combined), torch.cos(vec_combined)], dim=-1)
                rgb_combined = rgb_tensor.repeat(2, 1, 1, 1)

                with torch.inference_mode():
                    autocast_ctx = torch.autocast("cuda") if self.device == "cuda" else nullcontext()
                    with autocast_ctx:
                        preds = self.model(
                            x0=rgb_combined, # Changed to match infer.py
                            prompt=["predict glass segmentation", "predict depth map"],
                            task_emb=task_emb_combined,
                            timesteps=[self.timestep],
                            output_type="np",
                            processing_res=self.processing_res,
                            match_input_res=True,
                        ).images

                pred_seg = preds[0]
                pred_depth = preds[1]
                depth_npy = pred_depth.mean(axis=-1)
                
                # Segmentation
                p_min, p_max = pred_seg.min(), pred_seg.max()
                if p_max - p_min > 0.1:
                    pred_seg = (pred_seg - p_min) / (p_max - p_min)

                pred_gray = np.max(pred_seg, axis=-1)
                pred_mask = (pred_gray > self.threshold).astype(np.uint8) * 255

                # Overlay
                overlay = rgb_np.copy()
                color = np.array([255, 0, 0], dtype=np.uint8)
                alpha = 0.4
                mask_bool = pred_mask.astype(bool)
                overlay[mask_bool] = ((1 - alpha) * overlay[mask_bool] + alpha * color).astype(np.uint8)

                kernel = np.ones((9, 9), np.uint8)
                border_mask = (cv2.dilate(pred_mask, kernel, iterations=1) > 0) & ~mask_bool
                overlay[border_mask] = [0, 255, 0]

                # Splice logic using imported methods
                background_mask = pred_mask == 0
                glass_mask = pred_mask > 0
                
                mixed_depth = gt_np.copy().astype(np.float32)
                valid_mask = (gt_np > 1e-3) & (depth_npy > 1e-3) & background_mask

                gt_disparity, gt_non_neg_mask = depth2disparity(depth=gt_np, return_mask=True)
                pred_non_neg_mask = depth_npy > 0
                valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

                disparity_pred, _ = align_depth_least_square(
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

                # Publishers
                Z_mm_clipped = np.clip(mixed_depth * 1000.0, 0, 65535).astype(np.uint16)

                out_rgb_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="rgb8")
                out_rgb_msg.header = msg_rgb.header 

                out_depth_msg = self.bridge.cv2_to_imgmsg(Z_mm_clipped, encoding="16UC1")
                out_depth_msg.header = msg_depth.header

                # Fast numpy PC2 generation (Maintained from original)
                fx, fy = self.cam_matrix[0, 0], self.cam_matrix[1, 1]
                cx, cy = self.cam_matrix[0, 2], self.cam_matrix[1, 2]

                step = self.pc_step
                depth_decimated = mixed_depth[::step, ::step]
                overlay_decimated = overlay[::step, ::step]

                H, W = depth_decimated.shape
                v, u = np.indices((H, W))
                u, v = u * step, v * step

                valid_mask_pc = depth_decimated > 0.001
                Z_pc = depth_decimated[valid_mask_pc]
                u_valid = u[valid_mask_pc]
                v_valid = v[valid_mask_pc]

                X = (u_valid - cx) * Z_pc / fx
                Y = (v_valid - cy) * Z_pc / fy

                colors = overlay_decimated[valid_mask_pc]
                r, g, b = colors[:, 0].astype(np.uint32), colors[:, 1].astype(np.uint32), colors[:, 2].astype(np.uint32)
                
                rgb_packed = (r << 16) | (g << 8) | b
                rgb_float = rgb_packed.view(np.float32)

                points = np.column_stack((X, Y, Z_pc, rgb_float))
                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
                ]

                out_pc_msg = pc2.create_cloud(msg_depth.header, fields, points)

                self.pub_rgb.publish(out_rgb_msg)
                self.pub_depth.publish(out_depth_msg)
                self.pub_pc.publish(out_pc_msg)

            except Exception as e:
                self.get_logger().error(f"Failed to process frame: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SilicaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()