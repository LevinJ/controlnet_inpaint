import cv2
import os
import glob
from pathlib import Path
import numpy as np
import json

# Classes
class_id_to_name = {
    0: "Actuator_ok",
    1: "Actuator_breakage",
    2: "Actuator_pin_ok",
    3: "Actuator_pin_breakage",
    4: "Piston_ok",
    5: "Piston_oil",
    6: "Piston_breakage",
    7: "Screw_ok",
    8: "Screw_untightened",
    9: "Guiderod_ok",
    10: "Guiderod_oil",
    11: "Guiderod_malposed",
    12: "surface_scratch",
    13: "Spring_ok",
    14: "Spring_variant",
    15: "Marker_ok",
    16: "Marker_breakage",
    17: "Line_ok",
    18: "Line_unaligned",
    19: "Exhaust_screw_ok",
    20: "Exhaust_screw_abnormal",
    21: "Support_surface_ok",
    22: "Support_surface_scratch"
}
#names:
#  0: Actuator_ok
#  1: Actuator_breakage
#  2: Actuator_pin_ok
#  3: Actuator_pin_breakage
#  4: Piston_ok
#  5: Piston_oil
#  6: Piston_breakage
#  7: Screw_ok
#  8: Screw_untightened
#  9: Guiderod_ok
#  10: Guiderod_oil
#  11: Guiderod_malposed
#  12: surface_scratch
#  13: Spring_ok
#  14: Spring_variant
#  15: Marker_ok
#  16: Marker_breakage
#  17: Line_ok
#  18: Line_unaligned
#  19: Exhaust_screw_ok
#  20: Exhaust_screw_abnormal
#  21: Support_surface_ok
#  22: Support_surface_scratch

class inpaint_prompt(object):
    def __init__(self):
        pass

    def gen_inpaint_prompt(self, images_dir, labels_dir, output_dir, select_class):
        """
        直接处理标注文件进行可视化

        Args:
            images_dir: 图像目录
            labels_dir: 标注文件目录
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 支持的图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

        # 获取所有图像文件
        image_files = []
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_dir, extension)))

        print(f"找到 {len(image_files)} 张图像")
        # 对图像文件进行排序
        image_files.sort()

        # 用于收集所有json_item
        json_data = []

        # 处理每张图像
        for image_path in image_files:
            self.process_image_and_labels(image_path, labels_dir, select_class, json_data)

        # 保存json数据到文件
        json_save_path = os.path.join(output_dir, f"inpaint_prompt_{self.view_angle}.json")
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"Json数据已保存到: {json_save_path}")

    def process_image_and_labels(self, image_path, labels_dir, select_class, json_data):
        print(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        assert image is not None, f"无法读取图像: {image_path}"

        img_height, img_width = image.shape[:2]
        label_path = os.path.join(labels_dir, Path(image_path).stem + ".txt")
        assert os.path.exists(label_path), f"未找到标注文件: {label_path}"

        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            class_id, x_center, y_center, width, height = map(float, line.split())
            if not int(class_id) in select_class:
                continue
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width - 1, x2)
            y2 = min(img_height - 1, y2)
            mask = [y1, y2, x1, x2]
            defect_name = class_id_to_name[int(class_id)]
            defect_name = defect_name.replace('_', ' ')
            prompt = f"{defect_name}, {self.view_angle} viewpoint"
            json_item = {
                "image_path": image_path,
                "mask": mask,
                "prompt": prompt,
                "class_id": int(class_id)
            }
            json_data.append(json_item)
        
        
       
    def run(self):
        # 使用方法
        # view_angle = 'bottom'
        # images_directory = f"./data/images/{view_angle}"  # 替换为你的图像目录
        # labels_directory = f"./data/labels/{view_angle}"  # 替换为你的标注文件目录
        view_angles = ['bottom', 'left', 'top', 'right', 'front', 'back']
        for view_angle in view_angles:
            images_directory = f"./data/21.02.2025/{view_angle}"  # Replace with your image directory
            labels_directory = f"./data/21.02.2025/label/21.02.2025-txt/{view_angle}"  # Replace with your annotation file directory
            save_directory = f"./data/data_save"
            select_class = np.arange(0, 23).tolist()  # 选择所有类别

            self.view_angle = view_angle

            self.gen_inpaint_prompt(images_directory, labels_directory, save_directory, select_class)
        return

if __name__ == "__main__":
    inpaint = inpaint_prompt()
    inpaint.run()




