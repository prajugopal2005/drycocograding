"""
Script to convert COCO format dataset to folder structure for training
Organizes images by grade (Grade_A, Grade_B, Grade_C)
"""

import json
import os
import shutil
from pathlib import Path

def prepare_dataset_from_coco(coco_dir, output_dir):
    """
    Convert COCO format dataset to folder structure
    
    Args:
        coco_dir: Path to COCO dataset directory
        output_dir: Path to output directory for organized dataset
    """
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)
    
    # Process train, valid, and test splits
    for split in ['train', 'valid', 'test']:
        split_dir = coco_dir / split
        annotations_file = split_dir / '_annotations.coco.json'
        
        if not annotations_file.exists():
            print(f"⚠️ Annotations file not found: {annotations_file}")
            continue
        
        print(f"\n📂 Processing {split} split...")
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create category mapping
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        print(f"   Found categories: {list(categories.values())}")
        
        # Create image to annotations mapping
        image_annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(category_id)
        
        # Process each image with progress tracking
        total_images = len(coco_data['images'])
        processed = 0
        
        for img_info in coco_data['images']:
            img_id = img_info['id']
            img_filename = img_info['file_name']
            img_path = split_dir / img_filename
            
            if not img_path.exists():
                processed += 1
                continue
            
            # Get the category for this image
            if img_id in image_annotations:
                # Get the most common category if multiple annotations
                category_ids = image_annotations[img_id]
                category_id = max(set(category_ids), key=category_ids.count)
                category_name = categories.get(category_id, 'unknown')
                
                # Skip the parent category
                if category_name == 'copra-grade':
                    processed += 1
                    continue
                
                # Create output directory for this category and split
                output_category_dir = output_dir / split / category_name
                output_category_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy image to the appropriate folder
                output_path = output_category_dir / img_filename
                shutil.copy2(img_path, output_path)
            
            processed += 1
            # Show progress every 10 images or at completion
            if processed % 10 == 0 or processed == total_images:
                percentage = (processed / total_images) * 100
                print(f"   Progress: {processed}/{total_images} ({percentage:.1f}%)", end='\r')
        
        print(f"   ✅ Completed {split} split")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Dataset Organization Summary")
    print("=" * 60)
    for split in ['train', 'valid', 'test']:
        split_dir = output_dir / split
        if split_dir.exists():
            print(f"\n{split.upper()}:")
            for category_dir in sorted(split_dir.iterdir()):
                if category_dir.is_dir():
                    count = len(list(category_dir.glob('*.jpg')))
                    print(f"  - {category_dir.name}: {count} images")

def main():
    # Paths
    coco_dir = r"c:\Users\prajw\Downloads\cocograding\Deep-Learning-Project-master\copra.v1i.coco"
    output_dir = r"c:\Users\prajw\Downloads\cocograding\Deep-Learning-Project-master\dataset_organized"
    
    print("🥥 Coconut Copra Dataset Preparation")
    print("=" * 60)
    print(f"📁 Source: {coco_dir}")
    print(f"📁 Output: {output_dir}")
    
    print("=" * 60)
    
    # Prepare dataset
    prepare_dataset_from_coco(coco_dir, output_dir)
    
    print("\n✅ Dataset preparation complete!")
    print(f"📂 Organized dataset saved to: {output_dir}")

if __name__ == "__main__":
    main()
