import os
import json
from typing import List

class AnnotationMerger:
    def __init__(self):
        self.annotation_folder = '/media/levin/DATA/checkpoints/controlnet/src/ControlNet/data/data_save'
        self.view_angles = ['bottom', 'left', 'top', 'right', 'front', 'back']
        self.exclusion_list = [0, 2, 4, 7, 9, 13, 15, 17, 19, 21]
        self.output_file = os.path.join(self.annotation_folder, 'inpaint_prompt.json')

    def get_annotation_files(self) -> List[str]:
        files = []
        for angle in self.view_angles:
            file_path = os.path.join(self.annotation_folder, f'inpaint_prompt_{angle}.json')
            if os.path.exists(file_path):
                files.append(file_path)
        return files

    def merge_annotations(self):
        merged = []
        files = self.get_annotation_files()
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if item.get('class_id') not in self.exclusion_list:
                        assert " ok" not in item.get("prompt", ""), f'Found " ok" in prompt: {item.get("prompt")}'
                        merged.append(item)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Merged {len(merged)} annotation items into {self.output_file}")

if __name__ == "__main__":
    merger = AnnotationMerger()
    merger.merge_annotations()
