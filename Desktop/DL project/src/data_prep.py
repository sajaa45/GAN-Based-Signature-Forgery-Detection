import os
import shutil
import random
import cv2

#splitting 20% validation and 80% training
def move_images(source_dir, dest_dir, split_ratio=0.2):
    os.makedirs(dest_dir, exist_ok=True)
    
    images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    if not images:
        print(f"No images found in {source_dir}")
        return 0
    
    val_images = [f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))] if os.path.exists(dest_dir) else []
    
    if val_images:
        print(f" Validation directory already contains {len(val_images)} images!")
        print(f"   Skipping {source_dir} to avoid duplicate moves.")
        return 0
    
    # Shuffle images randomly
    random.shuffle(images)
    
    num_val = int(len(images) * split_ratio)
    val_images = images[:num_val]
    
    for img in val_images:
        src_path = os.path.join(source_dir, img)
        dest_path = os.path.join(dest_dir, img)
        shutil.move(src_path, dest_path)
        print(f"Moved: {img}")
    
    print(f"Moved {len(val_images)} images from {source_dir} to {dest_dir}")
    return len(val_images)

def count_images(directory):
    """Count the number of image files in a directory"""
    if not os.path.exists(directory):
        return 0
    images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return len(images)

#Resize → grayscale → normalize
def image_preprocessing(image_path):
    img=cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    # Resize
    img=cv2.resize(img,(64,64))
    #grayscale
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #normalize
    img=img/255.0
    return img
def preprocess_and_save_directory(source_dir, dest_dir):
    """Preprocess images and save them to a new directory"""
    os.makedirs(dest_dir, exist_ok=True)
    processed_count = 0
    
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} does not exist")
        return 0
    
    image_files = [f for f in os.listdir(source_dir) 
                   if os.path.isfile(os.path.join(source_dir, f))]
    
    for img_file in image_files:
        src_path = os.path.join(source_dir, img_file)
        processed_img = image_preprocessing(src_path)
        
        if processed_img is not None:
            # Save the preprocessed image
            dest_path = os.path.join(dest_dir, img_file)
            # Denormalize back to 0-255 range for saving
            img_to_save = (processed_img * 255).astype('uint8')
            cv2.imwrite(dest_path, img_to_save)
            processed_count += 1
    
    print(f"Processed and saved {processed_count} images to {dest_dir}")
    return processed_count
def main():
    source_dirs = {
        "genuine": os.path.join("data", "train", "genuine"),
        "forge": os.path.join("data", "train", "forge")
    }
    
    val_dirs = {
        "genuine": os.path.join("data", "val", "genuine"),
        "forge": os.path.join("data", "val", "forge")
    }
    
    print("=== INITIAL COUNTS ===")
    for category in ["forge", "genuine"]:
        train_count = count_images(source_dirs[category])
        val_count = count_images(val_dirs[category])
        print(f"{category.capitalize()}:")
        print(f"  Train: {train_count} images")
        print(f"  Val:   {val_count} images")
    
    total_val_images = sum(count_images(val_dirs[cat]) for cat in val_dirs)
    moved_counts = {"forge": 0, "genuine": 0}  # Initialize here
    
    if total_val_images > 0:
        print(f"\nWARNING: Validation folders already contain {total_val_images} images!")
        response = input("Do you want to move more images anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Skipping image moving step...")
        else:
            print("\n=== MOVING IMAGES ===")
            for category in ["forge", "genuine"]:
                print(f"\nProcessing {category} images...")
                moved_count = move_images(source_dirs[category], val_dirs[category])
                moved_counts[category] = moved_count
    else:
        print("\n=== MOVING IMAGES ===")
        for category in ["forge", "genuine"]:
            print(f"\nProcessing {category} images...")
            moved_count = move_images(source_dirs[category], val_dirs[category])
            moved_counts[category] = moved_count
    
    print("\n=== PREPROCESSING IMAGES ===")
    for split in ["train", "val"]:
        for category in ["genuine", "forge"]:
            source_directory = os.path.join("data", split, category)
            dest_directory = os.path.join("data", "preprocessed", split, category)
            processed_count = preprocess_and_save_directory(source_directory, dest_directory)
            print(f"{split}/{category}: {processed_count} images preprocessed")
    
    print("\n=== FINAL COUNTS ===")
    total_moved = 0
    for category in ["forge", "genuine"]:
        train_count = count_images(source_dirs[category])
        val_count = count_images(val_dirs[category])
        total_moved += moved_counts[category]
        
        print(f"{category.capitalize()}:")
        print(f"  Train: {train_count} images")
        print(f"  Val:   {val_count} images")
        if moved_counts[category] > 0:
            print(f"  Moved: {moved_counts[category]} images")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total images moved: {total_moved}")
    print(f"Split ratio: 80% train, 20% validation")

if __name__ == "__main__":
    main()