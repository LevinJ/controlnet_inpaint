import os
import glob
import json
from pathlib import Path
# Import class_id_to_name from data_explore2.py
from inpainting.data_explore2 import class_id_to_name
import cv2
import numpy as np

class TrainingSampleGenerator:
    def process_annotation_jsons(self, json_file_list):
        import shutil
        json_dir = os.path.join(self.output_dir, "temp/wuhan_fac")
        for json_path in json_file_list:
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            # Get mask_type from file name
            json_file = os.path.basename(json_path)
            mask_type = json_file.replace('_annotations.json', '')
            image_save_dir = os.path.join(json_dir, mask_type)
            mask_save_dir = os.path.join(json_dir, f"{mask_type}_mask")
            os.makedirs(image_save_dir, exist_ok=True)
            os.makedirs(mask_save_dir, exist_ok=True)
            for anno in annotations:
                image_path = anno['image_path']
                mask_coords = anno['mask']
                image_name = os.path.basename(image_path)
                # Copy image to image_save_dir
                shutil.copy(image_path, os.path.join(image_save_dir, image_name))
                # Generate mask image
                img = cv2.imread(image_path)
                assert img is not None, f"Failed to load image: {image_path}"
                mask_img = np.zeros_like(img)
                x1, x2, y1, y2 = mask_coords['x1'], mask_coords['x2'], mask_coords['y1'], mask_coords['y2']
                mask_img[int(y1):int(y2), int(x1):int(x2)] = (255, 255, 255)
                mask_name = f"{Path(image_name).stem}_mask.png"
                cv2.imwrite(os.path.join(mask_save_dir, mask_name), mask_img)
    def __init__(self, images_dir, labels_dir, anno_dict, output_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.anno_dict = anno_dict
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def collect_image_files(self):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        image_files.sort()
        return image_files

    def parse_label_file(self, label_path, mask_type):
        boxes = []
        if not os.path.exists(label_path):
            return boxes
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    if int(class_id) == mask_type:
                        boxes.append((x_center, y_center, width, height))
                except Exception:
                    continue
        return boxes

    def get_mask_coordinates(self, boxes, img_width, img_height):
        assert boxes is not None and len(boxes) > 0, "No bounding boxes found for the given mask type."
        # Find the box with the smallest x_center
        min_box = min(boxes, key=lambda b: b[0])
        x_center, y_center, width, height = min_box
        # Convert normalized to pixel coordinates
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        # Clamp to image size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width - 1, x2)
        y2 = min(img_height - 1, y2)
        return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}

    def generate(self):
        image_files = self.collect_image_files()
        created_json_files = []
        for mask_type, idx_list in self.anno_dict.items():
            # Convert mask_type to string name
            mask_type_str = class_id_to_name.get(mask_type, str(mask_type))
            save_dir = os.path.join(self.output_dir, "temp/wuhan_fac")
            os.makedirs(save_dir, exist_ok=True)
            annotations = []
            for idx in idx_list:
                assert idx < len(image_files), f"Index {idx} out of range for image files list."
                image_path = image_files[idx]
                label_path = os.path.join(self.labels_dir, Path(image_path).stem + ".txt")
                # Get image size
                img = cv2.imread(image_path)
                assert img is not None, f"Failed to load image: {image_path}"
                img_height, img_width = img.shape[:2]
                boxes = self.parse_label_file(label_path, mask_type)
                mask_coords = self.get_mask_coordinates(boxes, img_width, img_height)
                annotations.append({
                    'image_path': os.path.abspath(image_path),
                    'mask_type': mask_type_str,
                    'mask': mask_coords
                })
            # Save annotation json
            json_path = os.path.join(save_dir, f"{mask_type_str}_annotations.json")
            with open(json_path, 'w') as f:
                json.dump(annotations, f, indent=2)
            created_json_files.append(json_path)
        # Call the new method to process annotation jsons
        self.process_annotation_jsons(created_json_files)
        

if __name__ == "__main__":
    images_directory = "./data/images/bottom"
    labels_directory = "./data/labels/bottom"
    anno_dict = {11: [17, 47, 69], 10: [6, 11, 15]}
    output_dir = os.path.dirname(os.path.abspath(__file__))
    generator = TrainingSampleGenerator(images_directory, labels_directory, anno_dict, output_dir)
    generator.generate()
