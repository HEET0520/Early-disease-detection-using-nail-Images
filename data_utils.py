import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import random
import logging
from ultralytics import YOLO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLO model globally for detection (from cell 11)
try:
    yolo_model = YOLO('yolov8n.pt')
except Exception as e:
    logger.warning(f"Could not load YOLO model. Detection will fail. Error: {e}")

# --- Augmentation Functions (from cell 5) ---

def augment_images(folder_path, target_count=350):
    images = [img for img in os.listdir(folder_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
    current_count = len(images)

    if current_count >= target_count:
        print(f"Skipping {folder_path}, already has {current_count} images.")
        return

    num_augmented_needed = target_count - current_count
    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2),
        A.CLAHE(p=0.2)
        # Note: Removed ToTensorV2() from here, as we save back to jpg
    ])

    for _ in tqdm(range(num_augmented_needed), desc=f"Augmenting {os.path.basename(folder_path)}"):
        img_name = np.random.choice(images)
        img_path = os.path.join(folder_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"Could not read augmentation source: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = augmentations(image=image)['image']
        
        # Convert back to BGR for saving with cv2
        augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        new_name = f"{os.path.splitext(img_name)[0]}_aug{_}.jpg"
        new_path = os.path.join(folder_path, new_name)
        cv2.imwrite(new_path, augmented_bgr)

    print(f"Augmented {num_augmented_needed} images in {folder_path}")


def augment_folders(train_dir, min_threshold=300, target_count=350):
    class_folders = [os.path.join(train_dir, folder) for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]

    for class_folder in class_folders:
        images = [img for img in os.listdir(class_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]
        if len(images) < min_threshold:
            augment_images(class_folder, target_count)

# --- Data Reduction Functions (from cell 7) ---

def reduce_images(folder_path, target_count=350):
    images = [img for img in os.listdir(folder_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
    current_count = len(images)

    if current_count <= target_count:
        print(f"Skipping {folder_path}, already has {current_count} images.")
        return

    num_to_delete = current_count - target_count
    images_to_delete = random.sample(images, num_to_delete)

    for img_name in images_to_delete:
        os.remove(os.path.join(folder_path, img_name))

    print(f"Deleted {num_to_delete} images in {folder_path}")

def reduce_folders(train_dir, max_threshold=400, target_count=350):
    class_folders = [os.path.join(train_dir, folder) for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]

    for class_folder in class_folders:
        images = [img for img in os.listdir(class_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]
        folder_name = os.path.basename(class_folder)

        if len(images) > max_threshold and folder_name.lower() != "clubbing":
            reduce_images(class_folder, target_count)

# --- Renaming Function (from cell 9) ---

def rename_images_in_folders(train_dir):
    class_folders = [folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]

    for class_folder in class_folders:
        folder_path = os.path.join(train_dir, class_folder)
        images = sorted(os.listdir(folder_path))

        for index, img_name in enumerate(images, start=1):
            old_path = os.path.join(folder_path, img_name)
            extension = os.path.splitext(img_name)[-1]
            new_name = f"{class_folder}_img{index}{extension}"
            new_path = os.path.join(folder_path, new_name)

            counter = 1
            while os.path.exists(new_path):
                new_name = f"{class_folder}_img{index}_{counter}{extension}"
                new_path = os.path.join(folder_path, new_name)
                counter += 1

            os.rename(old_path, new_path)

        print(f"Renamed images in folder: {class_folder}")


# --- Preprocessing & Detection Functions (from cell 14) ---

def detect_and_crop(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return None

        results = yolo_model(image)
        detections = results[0].boxes.xyxy.cpu().numpy()

        if len(detections) == 0:
            logger.info(f"No detection found in {image_path}, using full image")
            return image  # Return full image

        x1, y1, x2, y2 = detections[0]
        padding_x = int(0.1 * (x2 - x1))
        padding_y = int(0.1 * (y2 - y1))

        x1 = max(0, int(x1) - padding_x)
        y1 = max(0, int(y1) - padding_y)
        x2 = min(image.shape[1], int(x2) + padding_x)
        y2 = min(image.shape[0], int(y2) + padding_y)

        cropped = image[y1:y2, x1:x2]
        return cropped

    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None

def preprocess_image(image):
    try:
        if image is None:
            return None
        if image.shape[0] < 10 or image.shape[1] < 10:
            logger.warning(f"Image too small: {image.shape}")
            return None

        image = cv2.GaussianBlur(image, (5, 5), 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        
        if variance < 1000:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) * 1.5
        else:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        
        image = cv2.filter2D(image, -1, kernel)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        image = image / 255.0
        return (image * 255).astype(np.uint8)

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return None

def process_dataset(dataset_path, cp):
    """
    Runs the full detection and preprocessing pipeline on a dataset folder.
    (Based on cell 14)
    """
    stats = {
        'total_images': 0, 'processed_images': 0, 'failed_images': 0, 'categories': {}
    }
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            continue
            
        save_category_path = os.path.join(cp, category)
        os.makedirs(save_category_path, exist_ok=True)
        stats['categories'][category] = {'total': 0, 'processed': 0, 'failed': 0}
        
        image_files = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        stats['total_images'] += len(image_files)
        stats['categories'][category]['total'] = len(image_files)

        for image_name in tqdm(image_files, desc=f"Processing {category}"):
            image_path = os.path.join(category_path, image_name)
            cropped_nail = detect_and_crop(image_path)
            
            if cropped_nail is not None:
                preprocessed_image = preprocess_image(cropped_nail)
                if preprocessed_image is not None:
                    save_image_path = os.path.join(save_category_path, image_name)
                    cv2.imwrite(save_image_path, preprocessed_image)
                    stats['processed_images'] += 1
                    stats['categories'][category]['processed'] += 1
                else:
                    stats['failed_images'] += 1; stats['categories'][category]['failed'] += 1
            else:
                stats['failed_images'] += 1; stats['categories'][category]['failed'] += 1
    
    # Print statistics
    logger.info("\nProcessing Statistics:")
    logger.info(f"Total images: {stats['total_images']}")
    if stats['total_images'] > 0:
        logger.info(f"Successfully processed: {stats['processed_images']} ({(stats['processed_images']/stats['total_images']*100):.2f}%)")
        logger.info(f"Failed: {stats['failed_images']} ({(stats['failed_images']/stats['total_images']*100):.2f}%)")
    logger.info("\nCategory-wise statistics:")
    for category, cat_stats in stats['categories'].items():
        logger.info(f"\n{category}:")
        logger.info(f"  Total: {cat_stats['total']}")
        if cat_stats['total'] > 0:
            logger.info(f"  Processed: {cat_stats['processed']} ({(cat_stats['processed']/cat_stats['total']*100):.2f}%)")
            logger.info(f"  Failed: {cat_stats['failed']} ({(cat_stats['failed']/cat_stats['total']*100):.2f}%)")
    logger.info("Dataset processing complete.")


# --- Main execution block to run all data prep steps ---
if __name__ == "__main__":
    
    RAW_IMAGE_DIR = "data/raw_images" # <-- Point this to your Kaggle dataset
    PROCESSED_DIR = "data/processed"  # <-- This is where processed images will be saved
    
    print(f"Starting data balancing and preprocessing...")
    
    if not os.path.exists(RAW_IMAGE_DIR):
        print(f"Error: Raw image directory not found at '{RAW_IMAGE_DIR}'")
        print("Please download your dataset and place it in that folder.")
    else:
        # Run Augmentation (Cell 5)
        print("\n--- Running Augmentation ---")
        augment_folders(RAW_IMAGE_DIR, min_threshold=300, target_count=350)
        
        # Run Data Reduction (Cell 7)
        print("\n--- Running Data Reduction ---")
        reduce_folders(RAW_IMAGE_DIR, max_threshold=400, target_count=350)
        
        # Run Renaming (Cell 9)
        # print("\n--- Running Renaming ---")
        # rename_images_in_folders(RAW_IMAGE_DIR)
        print("\n--- Skipping Renaming ---")
        print("(Note: Renaming is commented out in data_utils.py. Uncomment if needed.)")
        
        # Run Preprocessing (Cell 14)
        print("\n--- Running Cropping & Preprocessing ---")
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        process_dataset(RAW_IMAGE_DIR, PROCESSED_DIR)
        
        print("\nAll data utility tasks complete.")