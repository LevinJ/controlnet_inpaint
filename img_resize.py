# import cv2
# import os
#
# source_path = '/home/vincent/Documents/Wuhan-Factory/ControlNet/training/fill50k/source'
# target_path = '/home/vincent/Documents/Wuhan-Factory/ControlNet/training/fill50k/target'
#
#
# source_dir = os.listdir(source_path)
# target_dir = os.listdir(target_path)
#
# for source_name in list(source_dir):
#     print('source name: {}'.format(source_name))
#     source = cv2.imread(source_path + '/' + source_name)
#     image0 = cv2.resize(source, (512, 512), interpolation=cv2.INTER_NEAREST)
#     cv2.imwrite(source_path + '/' + source_name, image0)
#
# for target_name in list(target_dir):
#     print('target name: {}'.format(target_name))
#     target = cv2.imread(target_path + '/' + target_name)
#     target2 = cv2.resize(target, (512, 512))
#     cv2.imwrite(target_path + '/' + target_name, target2)


import cv2
import os

source_path = '/home/vincent/Documents/Wuhan-Factory/ControlNet/training/fill50k/source'
target_path = '/home/vincent/Documents/Wuhan-Factory/ControlNet/training/fill50k/target'

source_dir = os.listdir(source_path)
target_dir = os.listdir(target_path)

for source_name in list(source_dir):
    img_path = source_path + '/' + source_name
    print('img path: {}'.format(img_path))
    imgs = os.listdir(img_path)
    for img in imgs:
        source = cv2.imread(img_path + '/' + img)
        image0 = cv2.resize(source, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(img_path + '/' + img, image0)

for target_name in list(target_dir):
    img_path = target_path + '/' + target_name
    print('img path: {}'.format(img_path))
    imgs = os.listdir(img_path)
    for img in imgs:
        source = cv2.imread(img_path + '/' + img)
        image0 = cv2.resize(source, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(img_path + '/' + img, image0)