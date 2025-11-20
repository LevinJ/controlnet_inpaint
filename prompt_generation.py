import os
import json
from pathlib import Path


def generate_image_pairs_advanced(source_root_dir, target_root_dir, output_json_path,
                                  prompt_template="{category} missing from {subcategory}"):
    """
    高级版本：支持自定义prompt模板

    Args:
        target_root_dir: target文件夹的根目录路径
        output_json_path: 输出的JSON文件路径
        source_dir_name: source文件夹名称
        prompt_template: prompt模板，可以使用 {category}, {subcategory}, {filename} 等变量
    """

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    image_pairs = []

    for target_root, target_dirs, target_files in os.walk(target_root_dir):
        for file in target_files:
            target_file_path = Path(target_root) / file
            if target_file_path.suffix.lower() in image_extensions:
                target_name = str(target_root_dir).split('/')[-1]
                target_relative_path = target_file_path.relative_to(target_root_dir)

                for source_root, source_dirs, source_files in os.walk(source_root_dir):
                    for file in source_files:
                        source_file_path = Path(source_root) / file
                        if source_file_path.suffix.lower() in image_extensions:
                            source_name = str(source_root_dir).split('/')[-1]
                            source_relative_path = source_file_path.relative_to(source_root_dir)

                            # 解析路径信息用于prompt
                            path_parts = target_relative_path.parts
                            path_info = {
                                'filename': source_file_path.stem,
                                'category': path_parts[0] if len(path_parts) > 0 else "unknown",
                                'subcategory': path_parts[1] if len(path_parts) > 1 else "unknown",
                                'full_path': str(target_relative_path)
                            }

                            # 生成prompt
                            try:
                                prompt = prompt_template.format(**path_info)
                            except KeyError:
                                prompt = f"{path_info['category']} missing from {path_info['subcategory']}"

                            # 创建条目
                            pair = {
                                "source": str(Path("source" +'/' + source_name) / source_relative_path),
                                "target": str(Path("target" +'/' + target_name) / target_relative_path),
                                "prompt": prompt
                            }

                            image_pairs.append(pair)


    # 保存结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        for pair in image_pairs:
            json_line = json.dumps(pair, ensure_ascii=False)
            f.write(json_line + '\n')

    return image_pairs


# 使用示例
if __name__ == "__main__":
    # 基本用法
    # pairs = generate_image_pairs_advanced("target", "image_pairs.json")
    # print(f"生成了 {len(pairs)} 个图片对")

    # 自定义prompt模板的用法
    custom_prompt = "screw untightened from bottom angle"

    output_file = "/home/vincent/Documents/Wuhan-Factory/ControlNet/training/fill50k/prompt_bottom_screw_untightened.json"

    source_dir = '/home/vincent/Documents/Wuhan-Factory/ControlNet/training/fill50k/source/bottom_screw_ok'

    target_dir = "/home/vincent/Documents/Wuhan-Factory/ControlNet/training/fill50k/target/bottom_Screw_untightened"


    pairs_custom = generate_image_pairs_advanced(
        source_root_dir=source_dir,
        target_root_dir=target_dir,
        output_json_path=output_file,
        prompt_template=custom_prompt
    )