import torch
from pathlib import Path
from transformers import AutoModel
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import warnings
warnings.filterwarnings("ignore")

class ImageDataset(Dataset):
    def __init__(self, image_paths, data_base_path, transform):
        self.image_paths = image_paths
        self.data_base_path = Path(data_base_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_file = self.image_paths[idx]
        img_path = self.data_base_path / img_file
        if not img_path.exists():
            # This print can be noisy with many workers if many files are missing
            # print(f"Warning: {img_path} not found") 
            return None, None

        try:
            image = Image.open(img_path).convert('RGB')
            # Process with pad_square strategy to match your data loader
            width, height = image.size
            max_dim = max(width, height)
            new_image = Image.new('RGB', (max_dim, max_dim), color=(0, 0, 0))
            paste_x = (max_dim - width) // 2
            paste_y = (max_dim - height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image.resize((224, 224), Image.LANCZOS)
            
            return self.transform(image), img_file
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None, None

def collate_fn(batch):
    # Filter out None entries, which happen if an image file is missing or corrupt
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
        
    images = torch.stack([item[0] for item in batch])
    paths = [item[1] for item in batch]
    return images, paths

def extract_dino_features(
    unique_images_path="unique_images.txt",
    data_base_path="/workspace/vaani_data", 
    output_base_path="/workspace/cached_features/dinov2_large",
    batch_size=128,
    skip_existing=True,
    num_workers=None
):
    if num_workers is None:
        num_workers = 8#os.cpu_count()

    # Load the model once
    model = AutoModel.from_pretrained('facebook/dinov2-large').cuda()
    model.eval()
    
    # Your standard transform WITHOUT augmentations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load unique images
    with open(unique_images_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(image_paths)} unique images")
    
    Path(output_base_path).mkdir(parents=True, exist_ok=True)
    
    # Filter out already processed if skip_existing
    if skip_existing:
        remaining = []
        for img_file in image_paths:
            output_path = Path(output_base_path) / img_file.replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.png', '.pt')
            if not output_path.exists():
                remaining.append(img_file)
        print(f"Skipping {len(image_paths) - len(remaining)} already processed, {len(remaining)} to go")
        image_paths = remaining

    if not image_paths:
        print("No new images to process.")
        return

    dataset = ImageDataset(image_paths, data_base_path, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    with torch.no_grad():
        for batch_tensor, valid_paths in tqdm(dataloader, total=len(dataloader)):
            if batch_tensor is None:
                continue
                
            # Move to GPU
            batch_tensor = batch_tensor.cuda(non_blocking=True)
            features = model(batch_tensor).last_hidden_state
            
            # Save each individually
            for feat, img_file in zip(features, valid_paths):
                output_path = Path(output_base_path) / img_file.replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.png', '.pt')
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feat.cpu().to(torch.bfloat16), output_path)

if __name__ == "__main__":
    extract_dino_features()