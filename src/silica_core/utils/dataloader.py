import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
import numpy as np

class Trans10kDataset(Dataset):
    def __init__(self, base_dir, split, difficulty=None, resize=True, resolution=768):
        self.root_dir = Path(base_dir)
        self.resize = resize
        self.split = split

        if split == "test":
            if difficulty is None:
                raise ValueError("For test split, you must specify difficulty='easy' or 'hard'")
            
            self.image_dir = self.root_dir / "test" / difficulty / "image"
            self.mask_dir  = self.root_dir / "test" / difficulty / "mask"
        else:
            self.image_dir = self.root_dir / split / "image"
            self.mask_dir  = self.root_dir / split / "mask"
        
        if self.resize:
            self.img_transform = T.Compose([
                T.Resize((resolution, resolution)),
                T.ToTensor(),
                T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])
        else:
            self.img_transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])

        self.mask_resize = T.Resize(
            (resolution, resolution),
            interpolation=T.InterpolationMode.NEAREST
        )

        self.image_map = {
            p.stem: p
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".png",".jpg"}
        }

        self.mask_map = {}
        for p in self.mask_dir.iterdir():
            if p.suffix.lower() not in {".png",".jpg"}:
                continue

            stem = p.stem
            if stem.endswith("_mask"):
                stem = stem[:-5]

            self.mask_map[stem] = p

        self.keys = sorted(set(self.image_map.keys()) & set(self.mask_map.keys()))

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]

        img_path = self.image_map[key]
        mask_path = self.mask_map[key]

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        image = self.img_transform(image) 

        if self.resize:
            mask = self.mask_resize(mask)

        mask = TF.to_tensor(mask)          
        mask = (mask > 0.5).float()
        mask = mask * 2.0 - 1.0
        mask = mask.repeat(3,1,1)

        output = {
            "pixel_values": image,
            "seg_values": mask
        }

        if self.split == "test":
            output["filename"] = img_path.name

        return output


class GSDSDataset(Dataset):
    def __init__(self, base_dir, split, resize=True, resolution=768):
        self.resize = resize        
        self.split = split
        self.root_dir = Path(base_dir)
        self.image_dir = self.root_dir / split / "images"
        self.mask_dir = self.root_dir / split / "masks"
        
        if self.resize:
            self.img_transform = T.Compose([
                T.Resize((resolution, resolution)),
                T.ToTensor(),
                T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])
        else:
            self.img_transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])

        self.mask_resize = T.Resize(
            (resolution, resolution),
            interpolation=T.InterpolationMode.NEAREST
        )

        self.image_map = {p.stem : p for p in self.image_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg'}}
        self.mask_map = {p.stem: p for p in self.mask_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg'}}
                
        self.keys = sorted(set(self.image_map.keys()) & set(self.mask_map.keys()))

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]

        img_path = self.image_map[key]
        mask_path = self.mask_map[key]

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        image = self.img_transform(image) 

        if self.resize:
            mask = self.mask_resize(mask)

        mask = TF.to_tensor(mask)          
        mask = (mask > 0.5).float()
        mask = mask * 2.0 - 1.0
        mask = mask.repeat(3,1,1)

        output = {
            "pixel_values": image,
            "seg_values": mask
        }

        if self.split == "test":
            output["filename"] = img_path.name

        return output


class GDDDataset(Dataset):
    def __init__(self, base_dir, split, resize=True, resolution=768):
        self.resize = resize        
        self.split = split
        self.root_dir = Path(base_dir)
        self.image_dir = self.root_dir / split / "image"
        self.mask_dir = self.root_dir / split / "mask"
        
        if self.resize:
            self.img_transform = T.Compose([
                T.Resize((resolution, resolution)),
                T.ToTensor(),
                T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])
        else:
            self.img_transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])

        self.mask_resize = T.Resize(
            (resolution, resolution),
            interpolation=T.InterpolationMode.NEAREST
        )

        self.image_map = {p.stem : p for p in self.image_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg'}}
        self.mask_map = {p.stem: p for p in self.mask_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg'}}
                
        self.keys = sorted(set(self.image_map.keys()) & set(self.mask_map.keys()))

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]

        img_path = self.image_map[key]
        mask_path = self.mask_map[key]

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        image = self.img_transform(image)  

        if self.resize:
            mask = self.mask_resize(mask)

        mask = TF.to_tensor(mask)          
        mask = (mask > 0.5).float()
        mask = mask * 2.0 - 1.0
        mask = mask.repeat(3,1,1)

        output = {
            "pixel_values": image,
            "seg_values": mask
        }

        if self.split == "test":
            output["filename"] = img_path.name[:-4]

        return output


class Ref3DDataset(Dataset):
    def __init__(self, base_dir, split, resize=True, resolution=768):
        self.resize = resize        
        self.split = split
        self.root_dir = Path(base_dir)
        self.image_dir = self.root_dir / split / "image"
        self.mask_dir = self.root_dir / split / "mask"
        
        if self.resize:
            self.img_transform = T.Compose([
                T.Resize((resolution, resolution)),
                T.ToTensor(),
                T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])
        else:
            self.img_transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])

        self.mask_resize = T.Resize(
            (resolution, resolution),
            interpolation=T.InterpolationMode.NEAREST
        )

        self.image_map = {p.stem : p for p in self.image_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg'}}
        self.mask_map = {p.stem: p for p in self.mask_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg'}}
                
        self.keys = sorted(set(self.image_map.keys()) & set(self.mask_map.keys()))

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]

        img_path = self.image_map[key]
        mask_path = self.mask_map[key]

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        image = self.img_transform(image)  

        if self.resize:
            mask = self.mask_resize(mask)

        mask = TF.to_tensor(mask)          
        mask = (mask > 0.5).float()
        mask = mask * 2.0 - 1.0
        mask = mask.repeat(3,1,1)

        output = {
            "pixel_values": image,
            "seg_values": mask
        }

        if self.split == "test":
            output["filename"] = img_path.name

        return output
    

class Mirage18k(Dataset):
    def __init__(self, base_dir, split="train", resize=True, resolution=768):
        assert split in ["train", "test", "depth", "validation"]
        self.resize = resize
        self.split = split
        self.base_dir = Path(base_dir)

        if self.split == "depth":
            img_txt  = self.base_dir / "depth_test_rs_rgb.txt"
            mask_txt = self.base_dir / "depth_test_binary_mask.txt"
            depth_rs_txt = self.base_dir / "depth_test_rs_gt_depth_image.txt"
            raw_rs_txt = self.base_dir / "depth_test_rs_raw_depth_image.txt"

        elif self.split == "validation":
            img_txt  = self.base_dir / "validation_rgb.txt"
            mask_txt = self.base_dir / "validation_masks.txt"
            depth_rs_txt = self.base_dir / "validation_depth.txt"

        else:
            assert split in ["test", "train"]
            img_txt  = self.base_dir / f"seg_{split}_images.txt"
            mask_txt = self.base_dir / f"seg_{split}_masks.txt"

        with open(img_txt) as f:
            image_paths = [self.base_dir / line.strip() for line in f if line.strip()]

        with open(mask_txt) as f:
            mask_paths = [self.base_dir / line.strip() for line in f if line.strip()]

        img_map  = {p.stem: p for p in image_paths}
        mask_map = {p.stem: p for p in mask_paths}
        
        img_keys  = set(img_map.keys())
        mask_keys = set(mask_map.keys())

        if self.split == "depth":
            with open(raw_rs_txt) as f:
                raw_rs_paths = [self.base_dir / line.strip() for line in f if line.strip()]

            raw_rs_map = {p.stem: p for p in raw_rs_paths}
            raw_rs_keys = set(raw_rs_map.keys())

            with open(depth_rs_txt) as f:
                depth_rs_paths = [self.base_dir / line.strip() for line in f if line.strip()]

            depth_rs_map = {p.stem: p for p in depth_rs_paths}
            depth_rs_keys = set(depth_rs_map.keys())

            common_keys = sorted(img_keys & mask_keys & depth_rs_keys & raw_rs_keys)

            self.depth_rs_paths = [depth_rs_map[k] for k in common_keys]
            self.raw_rs_paths = [raw_rs_map[k] for k in common_keys]

        elif self.split == "validation":
            with open(depth_rs_txt) as f:
                depth_rs_paths = [self.base_dir / line.strip() for line in f if line.strip()]

            depth_rs_map = {p.stem: p for p in depth_rs_paths}
            depth_rs_keys = set(depth_rs_map.keys())

            common_keys = sorted(img_keys & mask_keys & depth_rs_keys)

            self.depth_rs_paths = [depth_rs_map[k] for k in common_keys]

        else:
            common_keys = sorted(img_keys & mask_keys)
            self.depth_rs_paths = None

        self.image_paths = [img_map[k] for k in common_keys]
        self.mask_paths  = [mask_map[k] for k in common_keys]

        if self.resize:
            self.img_transform = T.Compose([
                T.Resize((resolution, resolution)),
                T.ToTensor(),
                T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])
        else:
            self.img_transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
            ])

        self.mask_resize = T.Resize(
            (resolution, resolution),
            interpolation=T.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path  = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        image = self.img_transform(image)

        if self.resize:
            mask = self.mask_resize(mask)

        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        mask = mask * 2.0 - 1.0
        mask = mask.repeat(3,1,1)

        output = {
            "pixel_values": image,
            "seg_values": mask
        }

        if self.split == "test":
            output["filename"] = img_path.name

        if self.split == "depth":
            depth_rs_path = self.depth_rs_paths[idx]
            raw_rs_path = self.raw_rs_paths[idx]
            mask_path = self.mask_paths[idx]

            mask = Image.open(mask_path).convert("L")         
            glass_mask = np.array(mask) > 0
            background_mask = np.array(mask) == 0
            
            depth_rs = np.array(Image.open(depth_rs_path)).astype(np.float32)
            depth_rs = depth_rs / 1000.0 

            raw_rs = np.array(Image.open(raw_rs_path)).astype(np.float32)
            raw_rs = raw_rs / 1000.0 
            
            output["depth_gt"] = depth_rs
            output["depth_raw"] = raw_rs
            output["glass_mask"] = glass_mask
            output["background_mask"] = background_mask
            output["filename"] = img_path.name

        elif self.split == "validation":
            depth_rs_path = self.depth_rs_paths[idx]
            mask_path = self.mask_paths[idx]

            mask = Image.open(mask_path).convert("L")         
            glass_mask = np.array(mask) > 0
            
            depth_rs = np.array(Image.open(depth_rs_path)).astype(np.float32)
            depth_rs = depth_rs / 1000.0 
            
            output["depth_gt"] = depth_rs

        return output


class GWDDataset(Dataset):
    def __init__(self, base_dir, split="depth", resize=True, resolution=768):
        assert split in ["train", "depth"], "This dataset is strictly configured for the 'train' or 'depth' split."
        self.resize = resize
        self.split = split
        self.base_dir = Path(base_dir)

        if self.split == "depth":
            img_txt       = self.base_dir / "test_images.txt"
            mask_txt      = self.base_dir / "test_binary_masks.txt"
            depth_raw_txt = self.base_dir / "depth_raw.txt"
            depth_gt_txt  = self.base_dir / "depth_gt.txt"

            with open(depth_raw_txt) as f:
                raw_paths = [self.base_dir / line.strip() for line in f if line.strip()]
            with open(depth_gt_txt) as f:
                gt_paths = [self.base_dir / line.strip() for line in f if line.strip()]
                
            raw_map = {p.stem: p for p in raw_paths}
            gt_map  = {p.stem: p for p in gt_paths}
            
        else:
            img_txt  = self.base_dir / "train_images.txt"
            mask_txt = self.base_dir / "train_binary_masks.txt"

        with open(img_txt) as f:
            image_paths = [self.base_dir / line.strip() for line in f if line.strip()]
        with open(mask_txt) as f:
            mask_paths = [self.base_dir / line.strip() for line in f if line.strip()]

        img_map  = {p.stem: p for p in image_paths}
        mask_map = {p.stem: p for p in mask_paths}

        if self.split == "depth":
            common_keys = sorted(
                set(img_map.keys()) & 
                set(mask_map.keys()) & 
                set(raw_map.keys()) & 
                set(gt_map.keys())
            )
            self.depth_raw_paths = [raw_map[k] for k in common_keys]
            self.depth_gt_paths  = [gt_map[k] for k in common_keys]
        else:
            common_keys = sorted(
                set(img_map.keys()) & 
                set(mask_map.keys())
            )

        self.image_paths = [img_map[k] for k in common_keys]
        self.mask_paths  = [mask_map[k] for k in common_keys]

        if self.resize:
            self.img_transform = T.Compose([
                T.Resize((resolution, resolution)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.img_transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.mask_resize = T.Resize(
            (resolution, resolution),
            interpolation=T.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path  = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # --- 1. Load and Transform Image ---
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        # --- 2. Load and Transform Segmentation Mask ---
        mask_pil = Image.open(mask_path).convert("L")
        
        if self.resize:
            mask_tensor = self.mask_resize(mask_pil)
        else:
            mask_tensor = mask_pil

        mask_tensor = TF.to_tensor(mask_tensor)
        mask_tensor = (mask_tensor > 0.5).float()
        mask_tensor = mask_tensor * 2.0 - 1.0
        mask_tensor = mask_tensor.repeat(3, 1, 1)

        output = {
            "pixel_values": image,
            "seg_values": mask_tensor,
        }

        if self.split != "train":
            output["filename"] = img_path.name

        if self.split == "depth":
            depth_raw_path = self.depth_raw_paths[idx]
            depth_gt_path  = self.depth_gt_paths[idx]

            mask_np = np.array(mask_pil)
            glass_mask = mask_np > 0
            background_mask = mask_np == 0
            
            depth_gt = np.array(Image.open(depth_gt_path)).astype(np.float32)
            depth_gt = depth_gt / 1000.0 

            depth_raw = np.array(Image.open(depth_raw_path)).astype(np.float32)
            depth_raw = depth_raw / 1000.0 

            output["depth_gt"] = depth_gt
            output["depth_raw"] = depth_raw
            output["glass_mask"] = glass_mask
            output["background_mask"] = background_mask

        return output