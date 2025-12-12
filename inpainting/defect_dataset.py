import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from annotator.util import resize_image
import matplotlib.pyplot as plt
from collections import Counter


class DefectDataset(Dataset):
    def __init__(self):
        self.data = []
        # with open('./training/fill50k/prompt.json', 'rt') as f:
        #     for line in f:
        #         self.data.append(json.loads(line))
        with open('./data/data_save/inpaint_prompt.json', 'r') as f:
            self.data = json.load(f)
        # Hardcoded unwanted prompts
        # unwanted_prompts = [
        #     'Guiderod ok, bottom viewpoint',
        #     'Guiderod malposed, bottom viewpoint'
        # ]
        # self.data = [entry for entry in self.data if entry.get('prompt') not in unwanted_prompts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # source_filename = item['image_path']
        target_filename = item['image_path']
        prompt = item['prompt']
        mask = item['mask']

        target = cv2.imread(target_filename)
        # Do not forget that OpenCV read images in BGR order.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        source = target.copy().astype(np.float32)

        y1, y2, x1, x2 = mask
        source[y1:y2, x1:x2, :] = -255.0

        # modification for self dataset
        # source = cv2.resize(source, (512, 512))
        # source = cv2.resize(source, (512, 512))
        image_resolution = 512
        source = resize_image(source, image_resolution)
        target = resize_image(target, image_resolution)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    
if __name__ == "__main__":
    dataset = DefectDataset()
    print("Dataset length:", len(dataset))
    sample = dataset[3]
    print("Sample keys:", sample.keys())
    print("JPG shape:", sample['jpg'].shape)
    print("TXT:", sample['txt'])
    print("HINT shape:", sample['hint'].shape)
    print("prompt:", sample['txt'])
    print("Displaying sample images...")

    # --- Prompt statistics ---
    prompt_counter = Counter()
    for i in range(len(dataset)):
        prompt = dataset[i]['txt']
        prompt_counter[prompt] += 1
    print("\nPrompt statistics:")
    print(f"Total unique prompts: {len(prompt_counter)}")
    for prompt, count in prompt_counter.most_common():
        print(f"'{prompt}': {count}")
    print()
    # --- End prompt statistics ---

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("JPG (target)")
    plt.imshow((sample['jpg'] + 1.0) / 2.0)  # Convert back to [0,1] for display
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("HINT (source)")
    src_img = sample['hint'].copy()
    src_img[src_img < 0] = 0  # Set masked areas to black for visualization
    plt.imshow(src_img)
    plt.axis('off')

    plt.show()

