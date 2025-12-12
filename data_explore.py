import re
import cv2
import os
import glob
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import Entry
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
    DISPLAY_W = 1280
    DISPLAY_H = 960
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
        self.selected_bbox_idx = None
        self.annotation_edit_box = None

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
        self.annotation_file = label_path  # Set annotation file path here
        self.annotations = []
        self.bbox_pixel_coords = []
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
                self.annotations.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                width_px = width * img_width
                height_px = height * img_height
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width - 1, x2)
                y2 = min(img_height - 1, y2)
                self.bbox_pixel_coords.append((x1, y1, x2, y2))
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
        img = self._annotate_image(self.image_files[idx])
        return img

    def run(self):
        self.root = tk.Tk()
        self.root.title("Annotation Visualizer")
        self.image_label = ttk.Label(self.root)
        self.image_label.pack(padx=10, pady=10)
        self.index_label = ttk.Label(self.root, text="")
        self.index_label.pack()
        self.info_label = ttk.Label(self.root, text="", font=("Arial", 10))
        self.info_label.pack(pady=2)

        btn_frame = ttk.Frame(self.root)
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
        self.skip_entry.bind('<Return>', lambda event: self._skip_to_image())

        self.image_label.bind('<Button-1>', self._on_canvas_click)
        self.root.bind('<Up>', lambda event: self._prev_image())
        self.root.bind('<Down>', lambda event: self._next_image())
        self._show_image()
        self.root.mainloop()

    def _on_canvas_click(self, event):
        # Map click coordinates from displayed image to original image size
        display_w, display_h = self.DISPLAY_W, self.DISPLAY_H
        img = self._get_annotated_image(self.current_index)
        if img is None:
            return
        img_h, img_w = img.shape[:2]
        scale_x = img_w / display_w
        scale_y = img_h / display_h
        click_x = int(event.x * scale_x)
        click_y = int(event.y * scale_y)
        for idx, (x1, y1, x2, y2) in enumerate(self.bbox_pixel_coords):
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                self.selected_bbox_idx = idx
                self._show_annotation_edit_box()
                return
        self.selected_bbox_idx = None
        self._hide_annotation_edit_box()
        # If click is not in any bbox, show original image in matplotlib window
        import matplotlib.pyplot as plt
        img_path = self.image_files[self.current_index]
        orig_img = cv2.imread(img_path)
        if orig_img is not None:
            img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            plt.figure("Original Image")
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.show()

    def _show_image(self):
        img = self._get_annotated_image(self.current_index)
        if img is None:
            messagebox.showerror("Error", f"Cannot read image: {self.image_files[self.current_index]}")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((self.DISPLAY_W, self.DISPLAY_H))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self.tk_img)
        self.image_label.image = self.tk_img
        self.index_label.config(text=f"Image {self.current_index + 1}/{self.total_images}")
        self.info_label.config(text=f"Path: {self.image_files[self.current_index]}")
        print(f"Showing image {self.current_index + 1}/{self.total_images}: {self.image_files[self.current_index]}")
        self._hide_annotation_edit_box()
        return

    def _show_annotation_edit_box(self):
        if self.annotation_edit_box:
            self.annotation_edit_box.destroy()
        ann = self.annotations[self.selected_bbox_idx]
        self.annotation_edit_box = Entry(self.root, width=40)
        self.annotation_edit_box.insert(0, ann)
        # Place the box below the image label
        self.annotation_edit_box.place(x=10, y=self.image_label.winfo_height() + 10)
        self.annotation_edit_box.bind('<Return>', self._on_edit_enter)
        self.annotation_edit_box.focus_set()

    def _hide_annotation_edit_box(self):
        if self.annotation_edit_box:
            self.annotation_edit_box.destroy()
            self.annotation_edit_box = None

    def _on_edit_enter(self, event):
        new_ann = self.annotation_edit_box.get().strip()
        # Validate annotation format
        try:
            parts = list(map(float, new_ann.split()))
            assert len(parts) == 5
        except Exception:
            messagebox.showerror("Error", "Invalid annotation format. Must be: class x y w h")
            return
        self.annotations[self.selected_bbox_idx] = new_ann
        self._hide_annotation_edit_box()
        self._save_annotations_to_file()
        self._show_image()

    def _save_annotations_to_file(self):
        # Save updated annotations back to file
        with open(self.annotation_file, 'w') as f:
            for ann in self.annotations:
                f.write(ann + '\n')

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
    view_angle = "left"
    # images_directory = f"./data/images/{view_angle}"  # Replace with your image directory
    # labels_directory = f"./data/labels/{view_angle}"  # Replace with your annotation file directory
    images_directory = f"./data/21.02.2025/{view_angle}"  # Replace with your image directory
    labels_directory = f"./data/21.02.2025/label/21.02.2025-txt/{view_angle}"  # Replace with your annotation file directory
    select_class = None  # Or set to an integer class id to filter
    visualizer_gui = AnnotationVisualizerGUI(images_directory, labels_directory, select_class)
    visualizer_gui.run()