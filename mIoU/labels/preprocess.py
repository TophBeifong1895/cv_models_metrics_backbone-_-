import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def check_instance():
    imgs_path = glob.glob('./gtFine/train/*/*instanceTrainIds.png')
    unique_ids = []
    for img_path in imgs_path:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        tmp_unique_ids = np.unique(img)
        for ids in tmp_unique_ids:
            if ids not in unique_ids:
                unique_ids.append(ids)
    print(unique_ids)
# [0, 42, 43, 46, 70, 50, 54, 66, 58, 62, 51]


def make_train_txt():
    raw_img_path = 'leftImg8bit/train/*/*.png'
    imgs_path = glob.glob(raw_img_path)

    txt_path = './train_multitask.txt'
    txt = open(txt_path, 'w')
    for img_path in imgs_path:
        print(img_path)
        # raw depth instance semantic
        data = img_path + ' ' \
               + img_path.replace('leftImg8bit', 'disparity') + ' ' + \
               img_path.replace('leftImg8bit', 'gtFine').replace('gtFine.png', 'gtFine_instanceTrainIds.png') + ' ' + \
               img_path.replace('leftImg8bit', 'gtFine').replace('gtFine.png', 'gtFine_labelTrainIds.png') + '\n'
        txt.write(data)

def check_semantic():
    imgs_path = glob.glob('./gtFine/train/*/*labelIds.png')
    unique_ids = []
    for img_path in imgs_path:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        plt.imshow(img)
        plt.show()
        tmp_unique_ids = np.unique(img)
        for ids in tmp_unique_ids:
            if ids not in unique_ids:
                unique_ids.append(ids)
    print(unique_ids)
# [0, 1, 2, 5, 7, 8, 9, 10, 11, 13, 255, 12, 18, 6, 14, 17, 3, 4, 15, 16]




if False:
    check_instance()

if True:
    check_semantic()

if False:
    make_train_txt()
