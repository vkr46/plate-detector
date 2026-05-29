import os
import cv2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def check_images_exist(folder):
    if not os.path.exists(folder) or not os.listdir(folder):
        raise FileNotFoundError(f"No images found in the directory: {folder}")

def prepare_dataset(train_folder):
    check_images_exist(train_folder)
    images = load_images_from_folder(train_folder)
    print(f"Loaded {len(images)} images from {train_folder}")
    return images

if __name__ == "__main__":
    train_folder = "plate_data/images/train"
    try:
        prepare_dataset(train_folder)
    except Exception as e:
        print(e)