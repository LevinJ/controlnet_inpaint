import re
import cv2
import os
import glob
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk

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


class AnnotationVisualizerGUI:
    def __init__(self, images_dir, labels_dir, select_class=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.select_class = select_class
        self.image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.class_colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        self.image_files = self._get_image_files()
        self.total_images = len(self.image_files)
        self.current_index = 0
        self.annotated_images = [None] * self.total_images

    def _get_image_files(self):
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        image_files.sort()
        return image_files

    def _annotate_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        img_height, img_width = image.shape[:2]
        label_path = os.path.join(self.labels_dir, Path(image_path).stem + ".txt")
        if not os.path.exists(label_path):
            return image
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                class_id, x_center, y_center, width, height = map(float, line.split())
                if self.select_class is not None and int(class_id) != self.select_class:
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
                color = self.class_colors[int(class_id) % len(self.class_colors)]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_id_to_name[int(class_id)]}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except Exception as e:
                continue
        return image

    def _get_annotated_image(self, idx):
        if self.annotated_images[idx] is None:
            img = self._annotate_image(self.image_files[idx])
            self.annotated_images[idx] = img
        return self.annotated_images[idx]

    def run(self):
        root = tk.Tk()
        root.title("Annotation Visualizer")
        self.image_label = ttk.Label(root)
        self.image_label.pack(padx=10, pady=10)
        self.index_label = ttk.Label(root, text="")
        self.index_label.pack()
        self.info_label = ttk.Label(root, text="", font=("Arial", 10))
        self.info_label.pack(pady=2)

        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=5)
        prev_btn = ttk.Button(btn_frame, text="Prev", command=self._prev_image)
        prev_btn.grid(row=0, column=0, padx=5)
        next_btn = ttk.Button(btn_frame, text="Next", command=self._next_image)
        next_btn.grid(row=0, column=1, padx=5)
        skip_label = ttk.Label(btn_frame, text="Skip to index:")
        skip_label.grid(row=0, column=2, padx=5)
        self.skip_entry = ttk.Entry(btn_frame, width=5)
        self.skip_entry.grid(row=0, column=3, padx=5)
        skip_btn = ttk.Button(btn_frame, text="Skip", command=self._skip_to_image)
        skip_btn.grid(row=0, column=4, padx=5)

        self.image_label.bind('<Button-1>', self._popup_matplotlib)
        root.bind('<Up>', lambda event: self._prev_image())
        root.bind('<Down>', lambda event: self._next_image())
        self.skip_entry.bind('<Return>', lambda event: self._skip_to_image())
        self._show_image()
        root.mainloop()

    def _popup_matplotlib(self, event):
        import matplotlib.pyplot as plt
        img = self._get_annotated_image(self.current_index)
        if img is None:
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.title(f"Image {self.current_index + 1}: {self.image_files[self.current_index]}")
        plt.axis('off')
        plt.show()

    def _show_image(self):
        img = self._get_annotated_image(self.current_index)
        if img is None:
            messagebox.showerror("Error", f"Cannot read image: {self.image_files[self.current_index]}")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((1280, 960))  # Double the previous size
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self.tk_img)
        self.image_label.image = self.tk_img
        self.index_label.config(text=f"Image {self.current_index + 1}/{self.total_images}")
        self.info_label.config(text=f"Path: {self.image_files[self.current_index]}")
        print(f"Showing image {self.current_index + 1}/{self.total_images}: {self.image_files[self.current_index]}")

    def _prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self._show_image()

    def _next_image(self):
        if self.current_index < self.total_images - 1:
            self.current_index += 1
            self._show_image()

    def _skip_to_image(self):
        try:
            idx = int(self.skip_entry.get()) - 1
            if 0 <= idx < self.total_images:
                self.current_index = idx
                self._show_image()
            else:
                messagebox.showwarning("Warning", "Index out of range.")
        except ValueError:
            messagebox.showwarning("Warning", "Please enter a valid integer.")


if __name__ == "__main__":
    images_directory = "./data/images/bottom"  # Replace with your image directory
    labels_directory = "./data/labels/bottom"  # Replace with your annotation file directory
    select_class = None  # Or set to an integer class id to filter
    visualizer_gui = AnnotationVisualizerGUI(images_directory, labels_directory, select_class)
    visualizer_gui.run()