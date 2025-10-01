import os
import shutil
import random

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
    if total_val_images > 0:
        print(f"\nWARNING: Validation folders already contain {total_val_images} images!")
        response = input("Do you want to continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            return
    
    print("\n=== MOVING IMAGES ===")
    moved_counts = {}
    for category in ["forge", "genuine"]:
        print(f"\nProcessing {category} images...")
        moved_count = move_images(source_dirs[category], val_dirs[category])
        moved_counts[category] = moved_count
    
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