import cv2
import os
import glob
from pathlib import Path
import numpy as np


def visualize_annotations_directly(images_dir, labels_dir, output_dir, select_class):
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

    # 类别颜色（可以根据需要修改）
    class_colors = [
        (255, 0, 0),  # 红色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 蓝色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 黄色
    ]

    # 处理每张图像
    for idx, image_path in enumerate(image_files):
        # 读取图像
        print(f"Processing image: {idx + 1}/{len(image_files)} {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue

        img_height, img_width = image.shape[:2]

        # 对应的标注文件路径
        label_path = os.path.join(labels_dir, Path(image_path).stem + ".txt")

        # 检查标注文件是否存在
        if not os.path.exists(label_path):
            print(f"未找到标注文件: {label_path}")
            continue

        # 读取标注文件
        with open(label_path, 'r') as f:
            lines = f.readlines()
        vis_img = False
        # 处理每个标注
        for line in lines:
            line = line.strip()
            if not line:
                continue

            try:
                # 解析标注：class_id, x_center, y_center, width, height
                class_id, x_center, y_center, width, height = map(float, line.split())

                # if int(class_id) in [1, 3, 5, 6, 8, 10, 11, 12, 14, 16, 18, 20, 22]:
                # if True:
                if int(class_id) == select_class:
                    # 转换为像素坐标
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height

                    # 计算边界框坐标
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    # 确保坐标在图像范围内
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width - 1, x2)
                    y2 = min(img_height - 1, y2)

                    # if x1 > img_width/2:
                    #     continue

                    # 选择颜色
                    color = class_colors[int(class_id) % len(class_colors)]

                    # 绘制边界框
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    # 添加类别标签
                    # label = f"Class {int(class_id)}"
                    label = f"{class_id_to_name[int(class_id)]}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                                  (x1 + label_size[0], y1), color, -1)
                    cv2.putText(image, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    cropped_image = image[y1:y2, x1:x2]
                    vis_img = True

                    # save_dir = output_dir + '/' + str(label)
                    # 保存可视化结果
                    # output_path = os.path.join(output_dir, Path(image_path).name)
                    # cv2.imwrite(output_path, image)
                    # cv2.imwrite(output_path, cropped_image)
                    # print(f"已保存: {output_path}")
                else:
                    pass
            
            except Exception as e:
                print(f"处理标注时出错 {label_path}: {e}")
                continue

        if vis_img:
            cv2.imshow("Annotated Image", image)
            cv2.waitKey(0)
    cv2.destroyAllWindows()


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

# 使用方法
images_directory = "./data/images/bottom"  # 替换为你的图像目录
labels_directory = "./data/labels/bottom"  # 替换为你的标注文件目录
save_directory = './data/data_save/1'
select_class = int(11)  # 替换为你想要可视化的类别ID

visualize_annotations_directly(images_directory, labels_directory, save_directory, select_class)