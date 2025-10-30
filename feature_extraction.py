import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# Import custom model
from models import WeightedAvgFeatureExtractor

# --- Configuration ---
PROCESSED_DIR = "data/processed" # Input: Where processed images are
OUTPUT_CSV = "data/extracted_features.csv" # Output: File to save features
BATCH_SIZE = 32

# --- Dataset Class (from cell 15) ---
class NailDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # Get sorted class names from directories
        self.class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        
        # Create a mapping from class name to label index
        self.class_to_label = {class_name: i for i, class_name in enumerate(self.class_names)}
        
        for class_name in self.class_names:
            label = self.class_to_label[class_name]
            class_path = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_path):
                # Ensure it's a valid image file
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_path}. Returning black image.")
            image = np.zeros((224, 224, 3), dtype=np.uint8) # Return placeholder
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx], img_path

# Define transformations (from cell 15)
feature_extraction_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def extract_features():
    print(f"Starting feature extraction from '{PROCESSED_DIR}'...")
    
    if not os.path.exists(PROCESSED_DIR):
        print(f"Error: Processed directory not found at '{PROCESSED_DIR}'")
        print("Please run 'python data_utils.py' first to process your images.")
        return

    # Set up dataset and dataloader
    dataset = NailDiseaseDataset(PROCESSED_DIR, transform=feature_extraction_transform)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Save the label mapping
    print(f"Found {len(dataset.class_names)} classes:")
    label_map = {i: name for name, i in dataset.class_to_label.items()}
    print(label_map)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the feature extraction model
    model = WeightedAvgFeatureExtractor().to(device)
    model.eval()

    all_features = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels, img_paths in tqdm(data_loader, desc="Extracting Features"):
            images = images.to(device)
            features = model(images).cpu().numpy()
            
            all_features.extend(features)
            all_labels.extend(labels.numpy())
            all_paths.extend(img_paths)

    print("Feature extraction complete. Converting to DataFrame...")

    # Use the folder names as labels for clarity
    label_names = [os.path.basename(os.path.dirname(p)) for p in all_paths]
    
    feature_df = pd.DataFrame({
        'image_path': all_paths,
        'label_name': label_names,
        'label_idx': all_labels # This is the 0-9 alphabetical index
    })

    feature_cols = [f'feature_{i}' for i in range(all_features[0].shape[0])]
    feature_array_df = pd.DataFrame(np.array(all_features), columns=feature_cols)

    final_df = pd.concat([feature_df, feature_array_df], axis=1)
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Features saved to {OUTPUT_CSV}")
    print(f"Total samples: {len(final_df)}, Feature dimension: {len(feature_cols)}")

if __name__ == "__main__":
    extract_features()