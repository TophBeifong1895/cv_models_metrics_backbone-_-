import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import time
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

id_dict = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

pred_result_list = [] # 存所有预测的结果

class Logger(object):
    def __init__(self, filename="Default.log", path="./"):
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, filename), "a", encoding='utf8',)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def getLog(log_root):
    log_root=os.path.abspath(log_root)
    if not os.path.exists(log_root):
        print(f'\033[31;43mwarning: log path is not exist: create new path <{log_root}> \033[0m')
        os.mkdir(log_root)
    log_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    sys.stdout = Logger(log_root+'//'+log_time+'.log')

def masks2segments(masks, strategy='largest'):
    # Convert masks(n,160,160) into segments(n,xy)
    segments = []
    for x in masks.int().cpu().numpy().astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype('float32'))
    return segments

def clip_segments(segments, shape):
    # Clip segments (xy1,xy2,...) to image shape (height, width)
    if isinstance(segments, torch.Tensor):  # faster individually
        segments[:, 0].clamp_(0, shape[1])  # x
        segments[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
        segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y

def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # segments[:, 0] -= pad[0]  # x padding
    # segments[:, 1] -= pad[1]  # y padding
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]  # width
        segments[:, 1] /= img0_shape[0]  # height
    return segments
# masks 转换为 实例四周的点 可视化


def val_cocoseg(seg_path,box_path):

    image_id = seg_path.split('//')[-1][:-4]

    image=cv2.imread(f"E:\\AXI_timetest\\time\\wcy\\Val\\{image_id}.jpg")

    with open(box_path,'r') as f:
        boxs = f.readlines()

    (h,w)=image.shape[:2]

    scale = min(640/h,640/w)

    masks = np.fromfile(seg_path,np.float32)

    if len(masks) != 0:
        n = int(len(masks)/(160*160))

        masks=masks.reshape(n,160,160)

        masks = torch.from_numpy(masks)


        masks = F.interpolate(masks[None], [640,640], mode='bilinear', align_corners=False)[0]  # CHW

        masks = masks.gt_(0.5)

        # 按照每个目标框 进行轮廓点求取
        for k in range(n):
            masks_2 = np.zeros((640,640))
            masks_1 = masks[k].squeeze(0).detach().numpy()
            # cv2.imshow(" ", masks_1)
            # cv2.waitKey(0)
            x = float(boxs[k].strip().split(' ')[1])
            y = float(boxs[k].strip().split(' ')[2])
            w1 =  float(boxs[k].strip().split(' ')[3])
            h1 =  float(boxs[k].strip().split(' ')[4])

            x1 = x*scale
            y1 = y*scale
            x2 = (x + w1)*scale
            y2 = (y + h1)*scale
            masks_2[int(y1):int(y2),int(x1):int(x2)] = masks_1[int(y1):int(y2),int(x1):int(x2)]
            # cv2.rectangle(masks_1,(int(x1),int(y1)),(int(x2),int(y2)),(255,255,255),2)
            # cv2.imshow(" ", masks_2)
            # cv2.waitKey(0)
            masks_3 = torch.from_numpy(masks_2).unsqueeze(0)
            segments = masks2segments(masks_3)
            segments = [
                scale_segments( [640,640], x, image.shape, normalize=True)
                for x in segments]

            pred_dict ={}
            segmentation = []
            data = boxs[k].strip().split(' ')
            pred_dict['image_id'] = int(image_id)
            pred_dict['category_id'] = id_dict[int(data[0])]
            pred_dict['score'] = float(data[5])
            pred_dict['bbox'] = [float(data[1]),float(data[2]),float(data[3]),float(data[4])]
            pred_dict['segmentation'] = []
            
            for s in segments[0]:
                segmentation.append(round(s[0]*w,2))
                segmentation.append(round(s[1]*h,2))
            pred_dict['segmentation'].append(segmentation)

            # 限制 mask 部分 轮廓点数量 大于3 
            if len(segmentation) < 7: 
                pass
            else:
                pred_result_list.append(pred_dict)
        
            # print(len(infile),len(segments))
            # for i in segments:
            #     for j in i:
            #         cv2.circle(image,(int(j[0]*w),int(j[1]*h)),3,(0,0,0),6)

            # cv2.imshow('',image)
            # cv2.waitKey()


if __name__ == "__main__":

    #  测试 yolo-seg 精度脚本 

    seg_source = R'E:\GitLab\metrics\compile_workspace\results_sim\yolov5s_seg_16_BY_pt\mask'   # 上板mask结果路径
    box_source = R'E:\GitLab\metrics\compile_workspace\results_sim\yolov5s_seg_16_BY_pt\box'    # 上板box结果路径
    json_path  = R'pred_result_16bit_BY_seg.json'

    log_root='./classlog'
    getLog(log_root)
    seg_list = os.listdir(seg_source)
    box_list = os.listdir(box_source)
    for seg in tqdm(seg_list):
        seg_path = seg_source + "//" + seg
        box_path = box_source + "//" + seg[:-4] + ".txt"
        val_cocoseg(seg_path,box_path)

    with open(json_path,'w') as f:
        out = json.dumps(pred_result_list,indent=4)
        f.write(out)

    anno_json = R'./instances_val2017.json'
    pred_json = json_path
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api

    eval = COCOeval(anno, pred, 'bbox') # segm bbox
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

    eval = COCOeval(anno, pred, 'segm') # segm bbox
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

