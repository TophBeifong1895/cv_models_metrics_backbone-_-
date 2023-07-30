'''
Author: your name
Date: 2022-03-11 16:04:07
LastEditTime: 2023-07-24 16:55:24
LastEditors: Fang Shuli 1499936320@qq.com
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \metircs_script\calAP.py
'''


'''
该脚本用于旋转框目标检测精度测试，该方法只是测旋转框目标检测精度的一种，仅供参考。
流程是将芯片推理的四个点坐标（obb格式）转换为hbb格式，即左上角坐标 x, y, 和 w, h。接下来的流程和coco测精度一致。
obb转换为hbb：{ class_id, point_x1, point_y1, point_x2, point_y2, point_x3,point_y3,point_x4,point_y4,score } -> {class_id, x, y, w, h, score}
注意事项：标注结果也需要进行obb转换为hbb。
'''

import sys
import os
import time
from unicodedata import name
import cv2
import json
import re
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import numpy as np

def poly2hbb(polys):
    """
    Trans poly format to hbb format
    Args:
        rboxes (array/tensor): (num_gts, poly) 

    Returns:
        hbboxes (array/tensor): (num_gts, [xc yc w h]) 
    """
    assert polys.shape[-1] == 8
    if isinstance(polys, torch.Tensor):
        x = polys[:, 0::2] # (num, 4) 
        y = polys[:, 1::2]
        x_max = torch.amax(x, dim=1) # (num)
        x_min = torch.amin(x, dim=1)
        y_max = torch.amax(y, dim=1)
        y_min = torch.amin(y, dim=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = torch.cat((x_ctr-w/2, y_ctr-h/2, w, h), dim=1)
    else:
        x = polys[:, 0::2] # (num, 4) 
        y = polys[:, 1::2]
        x_max = np.amax(x, axis=1) # (num)
        x_min = np.amin(x, axis=1) 
        y_max = np.amax(y, axis=1)
        y_min = np.amin(y, axis=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = np.concatenate((x_ctr-w/2, y_ctr-h/2, w, h), axis=1)
    return hbboxes


def yolores2cocores(originImagesDir, resultsDir, dtJsonPath):
    # originImageDir  : 存放测试集图片的目录                   ---> 对应C++中的img_root
    # resultsDir : 存放icrft输出txt（yolo格式）的目录     ---> 对应C++中的save_txt_path
    # savePath        : json文件的保存目录（包括json文件的名字）
    indexes = sorted(os.listdir(originImagesDir))
    dataset = []
    imgIds = []
    
    pattern = re.compile("^0+")

    # 标注的id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # 支持 png jpg 格式的图片。
        txtFile = index.replace('.jpg','.txt').replace('.png','.txt')
        try:
            img_id = int(re.sub(pattern, "", index).split('.')[0]) 
            # print("_id_try is : " + str(_id))
        except:
            _id = re.sub(pattern, "", index).split('.')[0]
            # print("_id_except is : " + str(_id))


            img_id = int(_id.split('_')[2])
            # img_id = _id

        # 添加图像的信息
        if not os.path.exists(os.path.join(resultsDir, txtFile)):
            # 如没标签，跳过，只保留图片信息。
            continue
        with open(os.path.join(resultsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split(' ')
                # class_id, point_x1, point_y1, point_x2, point_y2, point_x3,point_y3,point_x4,point_y4,score
                obb = torch.tensor([float(i) for i in label[1:9]]).unsqueeze(0)
                # obb = torch.tensor([float(i) for i in label[7:]]).unsqueeze(0)
                hbb = poly2hbb(obb).squeeze(0).tolist() # 将旋转框四个点坐标转换为xywh形式
                x1 = float(hbb[0])
                y1 = float(hbb[1])
                width = float(hbb[2])
                height = float(hbb[3])
            
                cls_id = int(label[0]) 
                score = float(label[9])
                # score = float(label[5])

                dataset.append({
                    #'area': width * height,
                    'image_id': img_id, # 这里如果直接用COCO的json，就不能是k，得是图片的名字（不含0）
                    'category_id': cls_id,
                    'bbox': [x1, y1, width, height], # coco要求传入相对于原图的左上角点坐标和wh
                    'score': score,
                    # 'id': ann_id_cnt, # id是检测框的id，
                    #'image_id': k,
                    # 'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    #'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1
                imgIds.append(img_id)
    # 保存结果

    with open(dtJsonPath, 'w') as f:
        json.dump(dataset, f)
        # print("f is : " + str(f))
        print('Save annotation to {}'.format(dtJsonPath))
    return imgIds

class Logger(object):
    def __init__(self, filename="Default.log", path="./"):
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, filename), "a", encoding='utf8',)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main(originImagesDir, resultsDir, gtJsonPath, dtJsonPath):
    # yolores2cocores
    # originImageDir  :  存放测试集图片的目录
    # resultsDir :  存放icraft输出txt（yolo格式）的目录
    # savePath        :  json文件的保存目录（包括json文件的名字）

    imgIds = yolores2cocores(originImagesDir, resultsDir, dtJsonPath)
    cocoGt=COCO(gtJsonPath)
    cocoDt=cocoGt.loadRes(dtJsonPath)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.imgIds = sorted(imgIds)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == "__main__":
    gtJsonPath = R"./gold_result_dior.json"    #json标签
    json_res_root = R"./results/" #推理结果json文件夹
    log_root = R"./log/" #计算结果log文件夹
    
    # 以下三项需要手动配置
    originImageDir = R"E:\GitLab\metrics\obb\test_acc\images" # 数据集路径
    txt_res_root = R"E:\GitLab\metrics\compile_workspace\results_onboard" # 推理结果父文件夹
    model_names = ['YoloV5_obb_dior_0724']    # 模型结果文件夹名字
    
    log_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    sys.stdout = Logger(log_root+log_time+'.log')                                                            #测试结果log记录地址
    for name in model_names:
        print('='*20+' '+name+' '+'='*20)
        resultsDir = txt_res_root + r"/{}/".format(name)
        dtJsonPath = json_res_root + r"/{}.json".format(name)
        print("dtJsonPath is : "+ str(dtJsonPath))
        main(originImageDir, resultsDir, gtJsonPath, dtJsonPath)
        print('='*50)