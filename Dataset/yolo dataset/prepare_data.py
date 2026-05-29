import os
import xml.etree.ElementTree as ET
import glob
import random
from tqdm import tqdm
import shutil

# --- CONFIGURATION ---
# Path to the unzipped Kaggle dataset
base_path = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(base_path, 'images')
annotations_path = os.path.join(base_path, 'annotations')

# Percentage split for train/validation
# 80% for training, 20% for validation
split_ratio = 0.8

# Output directory for YOLO formatted data (inside yolov7)
output_path = 'yolov7/plate_data'
# -------------------

def convert_xml_to_yolo(xml_file, image_width, image_height):
    """Converts a single XML annotation to YOLO .txt format"""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_annotations = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        # We only have one class: 'licence-plate', which will be class_id 0
        class_id = 0

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Convert to YOLO format (center_x, center_y, width, height) normalized
        dw = 1.0 / image_width
        dh = 1.0 / image_height

        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        x = center_x * dw
        y = center_y * dh
        w = width * dw
        h = height * dh

        yolo_annotations.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return yolo_annotations

def process_dataset():
    """Main function to process, convert, and split the dataset."""
    print("Starting dataset processing...")

    # Create output directories
    os.makedirs(os.path.join(output_path, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels/val'), exist_ok=True)

    xml_files = glob.glob(os.path.join(annotations_path, '*.xml'))
    random.shuffle(xml_files)

    split_index = int(len(xml_files) * split_ratio)
    train_files = xml_files[:split_index]
    val_files = xml_files[split_index:]

    print(f"Found {len(xml_files)} total annotations.")
    print(f"Splitting into {len(train_files)} training and {len(val_files)} validation files.")

    # Process files
    for split, files in [('train', train_files), ('val', val_files)]:
        print(f"\nProcessing {split} set...")
        for xml_file in tqdm(files):
            basename = os.path.basename(xml_file).replace('.xml', '')
            image_file_png = os.path.join(images_path, f"{basename}.png")

            # Check if image file exists
            if not os.path.exists(image_file_png):
                print(f"Warning: Image file not found for {basename}.xml, skipping.")
                continue

            # Get image dimensions from XML (more reliable than loading image)
            tree = ET.parse(xml_file)
            size = tree.getroot().find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)

            # Convert annotations
            yolo_data = convert_xml_to_yolo(xml_file, img_width, img_height)

            # Write YOLO label file
            label_output_path = os.path.join(output_path, f'labels/{split}/{basename}.txt')
            with open(label_output_path, 'w') as f:
                f.write('\n'.join(yolo_data))

            # Copy image file
            image_output_path = os.path.join(output_path, f'images/{split}/{basename}.png')
            shutil.copy(image_file_png, image_output_path)

    print("\nDataset processing complete!")
    print(f"Formatted data is now in: {os.path.abspath(output_path)}")

if __name__ == '__main__':
    process_dataset()