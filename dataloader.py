from dataclasses import asdict, dataclass
from torch.utils.data import Dataset
from torchvision import transforms
from utils import ResizeWithPadding
import os
import torch
from PIL import Image

@dataclass
class Sample:
    img: torch.Tensor = None
    img_path: str = ""
    mask: torch.Tensor = None
    mask_path: str = ""
    prompt: str = ""
    postive: bool = False
        

class OPMEDDataset(Dataset):
    def __init__(self, root_dir, modality='FLAIR', train=True, img_size=(256, 256)):
        self.root_dir = root_dir
        self.modality = modality
        self.train = train
        self.img_size = img_size
        self.transform = transforms.Compose([
            ResizeWithPadding(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        # Set paths based on train/test
        if train:
            self.hc_image_dir = os.path.join(root_dir, 'train', f'hc_{modality}')
            self.un_image_dir = os.path.join(root_dir, 'train', f'un_{modality}')
            self.mask_dir = None  # No masks for healthy controls
        else:
            self.hc_image_dir = os.path.join(root_dir, 'test', f'hc_{modality}')
            self.un_image_dir = os.path.join(root_dir, 'test', f'un_{modality}')
            self.mask_dir = os.path.join(root_dir, 'test', 'mask')
            
        self.hc_filenames = sorted([f for f in os.listdir(self.hc_image_dir) if f.endswith('.png')])
        self.un_filenames = sorted([f for f in os.listdir(self.un_image_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.hc_filenames) + len(self.un_filenames)
        
    def __getitem__(self, idx):   
        sample = Sample()
        image_dir = self.hc_image_dir
        filenames = self.hc_filenames
        is_positive = False
        
        if idx + 1 > len(self.hc_filenames):
            image_dir = self.un_image_dir
            filenames = self.un_filenames
            idx = idx % len(self.hc_filenames)
            is_positive = True
                 
        img_path = os.path.join(image_dir, filenames[idx])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("L")
            sample.img_path = img_path
        else:
            raise RuntimeError(f"sample {img_path} not found")
        
        if (not self.train) and is_positive:
            mask_filename = filenames[idx].replace(self.modality, f"{self.modality}_roi")
            mask_path = os.path.join(self.mask_dir, mask_filename)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                sample.mask_path = mask_path
        else:
            # Create a black mask (all zeros) with the same shape as the image
            mask = Image.new("L", (img.width, img.height), 0)
            
        # Apply transforms if specified
        if not self.transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])
            
        sample.img = self.transform(img)
        trans_mask = self.transform(mask)
        sample.mask = (trans_mask > 0.).float()
                
        return asdict(sample)
    

