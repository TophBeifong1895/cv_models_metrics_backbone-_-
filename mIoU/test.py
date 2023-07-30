import os
import cv2
import sys
import time
import logging
import argparse
import numpy as np
from util import config
from util.util import AverageMeter, intersectionAndUnion

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='./config.yaml', help='config file')
    # parser.add_argument('--config', type=str, default='E:\GitLab\metrics\mIoU\config.yaml', help='config file')
    parser.add_argument('opts', help='see config.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)

    data_list = make_dataset()

    index_start = args.index_start
    if args.index_step == 0:
        index_end = len(data_list)
    else:
        index_end = min(index_start + args.index_step, len(data_list))
    # print(index_end)
    data_list = data_list[index_start:index_end]

    # print('data_list is : ' + str(data_list))

    # with open(args.names_path) as f:
    with open(args.names_path) as f:
        names = f.read().splitlines()

    cal_acc(data_list, args.classes, names)

def cal_acc(data_list, classes, names):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i, (image_path, target_path) in enumerate(data_list):
        pred = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        # print("classes is : "+str(classes))
        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.info('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), os.path.split(image_path)[-1], accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    file_handler = logging.FileHandler('log_file.txt')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

    for i in range(classes):
        # print(i)
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))


def make_dataset():
    # Get label list from config.yaml
    labels_root = args.labels_root
    if args.labels_list is None:
        labels_list = check_file(labels_root, IMG_EXTENSIONS)
    else:
        # with open(args.labels_list, 'r') as f:
        with open(args.labels_list, 'r') as f:
            labels_list = f.read().splitlines()

    # Get result list from config.yaml
    results_root = args.results_root
    if args.results_list is None:
        results_list = check_file(results_root, IMG_EXTENSIONS)
    else:
        with open(args.results_list, 'r') as f:
            results_list = f.read().splitlines()
            # print(results_list)

    print("Totally {} prediction results in the folder.".format(len(results_list)))
    print("Totally {} ground truths in the folder.".format(len(labels_list)))
    # if len(results_list) != len(labels_list):
    #     raise NotImplementedError("Please check your prediction results and ground truths")
    print("Start checking image&label pair...")

    # Combine labels with results
    image_label_list = []
    for i in range(len(labels_list)):
        # print(results_list[i])
        image_name = os.path.join(results_root, results_list[i].strip())
        label_name = os.path.join(labels_root, labels_list[i].strip())
        # Check files
        if not os.path.isfile(image_name):
            raise FileNotFoundError("File: " + image_name + " does not exist")
        if not os.path.isfile(label_name):
            raise FileNotFoundError("File: " + label_name + " does not exist")

        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair done!")
    return image_label_list


def check_file(path, suffix_list):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))

    img_list = []
    for file in file_list:
        if any(file.lower().endswith(extension) for extension in suffix_list):
            img_list.append(file.strip())
    return img_list

if __name__ == '__main__':
    cv2.ocl.setUseOpenCL(False)
    main()
