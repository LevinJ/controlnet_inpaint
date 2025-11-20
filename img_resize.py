import cv2
import os

source_path = '/home/vincent/Documents/Wuhan-Factory/ControlNet/training/fill50k/source'
target_path = '/home/vincent/Documents/Wuhan-Factory/ControlNet/training/fill50k/target'

source_dir = os.listdir(source_path)
target_dir = os.listdir(target_path)

for source_name in list(source_dir):
    # img_path = source_path + '/' + source_name
    img_path = source_path
    print('img path: {}'.format(img_path))
    # imgs = os.listdir(img_path)

    sub_folder_names = os.listdir(img_path)
    for sub_folder_name in list(sub_folder_names):
        new_img_path = img_path + '/' + sub_folder_name
        imgs = os.listdir(new_img_path)
        for img in imgs:
            source = cv2.imread(new_img_path + '/' + img)
            # image0 = cv2.resize(source, (512, 512), interpolation=cv2.INTER_NEAREST)
            image0 = cv2.resize(source, (128, 128), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(new_img_path + '/' + img, image0)

for target_name in list(target_dir):
    # img_path = target_path + '/' + target_name
    img_path = target_path
    print('img path: {}'.format(img_path))

    # imgs = os.listdir(img_path)
    sub_folder_names = os.listdir(img_path)
    for sub_folder_name in list(sub_folder_names):
        new_target_path = img_path + '/' + sub_folder_name
        imgs = os.listdir(new_target_path)
        for img in imgs:
            source = cv2.imread(new_target_path + '/' + img)
            # image0 = cv2.resize(source, (512, 512), interpolation=cv2.INTER_NEAREST)
            image0 = cv2.resize(source, (128, 128), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(new_target_path + '/' + img, image0)