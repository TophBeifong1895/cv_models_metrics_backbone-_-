import os
import cv2
import json
import re
import sys
import time
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import Dict, List

def coco80_to_coco91_class(cat_id_80):  
    '''
    COCO80类别标签映射成91类, 若将80类标签下的数据送入coco则精度错误
    '''
    mapping = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return mapping[cat_id_80]


def convertOutputToCocoJson(originImagesDir : str,
                            inferenceOutputDir : str,
                            dtJsonPath : str,
                            is_norm : bool, 
                            is_center : bool, 
                            is_coco_80 : bool) -> List:
    '''
    将检测网络输出的txt文件转换成COCO支持的JSON文件并输出AP
    支持输出各类ap
    
    [input] 
    originImageDir       :    测试集图片的目录
    inferenceOutputDir   :    输出txt的目录
    dtJsonPath           :    生成JSON文件保存路径
    is_norm             
           true          :    检测网络输出归一化数据  
           false         :    检测网络输出非归一化数据
    is_center   
           true          :    检测网络输出物理含义(中心点x, 中心点y, 框宽w, 框高h)
           false         :    检测网络输出物理含义(左上角x, 左上角y, 框宽w, 框高h)
    is_coco_80
           true          :    检测网络输出类别为COCO 80类, 需要转成91类
           false         :    检测网络输出类别为COCO 91类, 无需转换
    [retval] 
    imgIds : 实际使用的测试图片id
    '''
    # indexes = sorted(os.listdir(originImagesDir))[0:100]
    # print(len(originImagesDir))
    indexes = sorted(os.listdir(originImagesDir))
    dataset = []
    imgIds = []
    pattern = re.compile("^0+")
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # 支持 png jpg 格式的图片
        txtFile = index.replace('.jpg','.txt').replace('.png','.txt')
        img_id = int(re.sub(pattern, "", index).split('.')[0])
        # 读取图像的宽和高
        im = cv2.imread(os.path.join(originImagesDir, index))
        height, width, _ = im.shape
        # 添加图像的信息
        if not os.path.exists(os.path.join(inferenceOutputDir, txtFile)):
            # 如没txt，表示没有检测该图片，跳过
            # print('ok')
            continue
        with open(os.path.join(inferenceOutputDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # TODO: 测试 
                H, W, _ = im.shape
                if is_center:
                    x1 = max(0, (x - w / 2) * W) if is_norm else max(0, x - w / 2)
                    y1 = max(0, (y - h / 2) * H) if is_norm else max(0, y - h / 2)
                    x2 = min(W, (x + w / 2) * W) if is_norm else min(W, x + w / 2)
                    y2 = min(H, (y + h / 2) * H) if is_norm else min(H, y + h / 2)
                else:
                    x1 = max(0, x * W) if is_norm else max(0, x) 
                    y1 = max(0, y * H) if is_norm else max(0, y)
                    x2 = min(W, (x + w) * W) if is_norm else min(W, x + w)
                    y2 = min(H, (y + h) * H) if is_norm else min(H, y + h)

                cls_id = int(coco80_to_coco91_class(int(label[0]))) if is_coco_80 else int(label[0])
                score = float(label[5])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                
                dataset.append({
                    #'area': width * height,
                    'image_id': img_id, # 这里如果直接用COCO的json，就不能是k，得是图片的名字（不含0）
                    'category_id': cls_id,
                    'bbox': [x1, y1, width, height],
                    'score': score,
                    # 'id': ann_id_cnt, # id是检测框的id，
                    # 'image_id': k,
                    # 'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    #'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1
            imgIds.append(img_id)
    # 保存结果

    with open(dtJsonPath, 'w') as f:
        json.dump(dataset, f)
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

def getLog(log_root):
    log_root=os.path.abspath(log_root)
    if not os.path.exists(log_root):
        print(f'\033[31;43mwarning: log path is not exist: create new path <{log_root}> \033[0m')
        os.mkdir(log_root)
    log_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    sys.stdout = Logger(log_root+'//'+log_time+'.log')

# main
def main():
    # originImagesDir、inferenceOutputDir、dtJsonPath需要手动修改路径
    originImagesDir = R"E:\AXI_timetest\time\wcy\Val"
    inferenceOutputDir = R"E:\GitLab\test_bench\test-dir\Temp_True202369_1738_16bits_channel\test_results\YoloV4_ori-16-channel-608x608"
    dtJsonPath = R"E:\GitLab\metrics\compile_workspace\results_json\YoloV4_16_pc.json"
    gtJsonPath = R"./instances_val2017_yolo.json"

    imgIds =  convertOutputToCocoJson(originImagesDir=originImagesDir, 
                                     inferenceOutputDir=inferenceOutputDir,
                                     dtJsonPath=dtJsonPath,
                                     is_norm=False, 
                                     is_center=False, 
                                     is_coco_80=False)
    log_root='./classlog'
    getLog(log_root)

    cocoGt=COCO(gtJsonPath)
    cocoDt=cocoGt.loadRes(dtJsonPath)

    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.imgIds = sorted(imgIds)
    # a = cocoEval.params.imgIds
    # b = imgIds
    # print((set(a)|set(b)) - (set(a)&set(b)))
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # ap per class 
    # precisions = cocoEval.eval['precision']
    # results_per_category = []
    # for idx, catId in enumerate(cocoGt.getCatIds()):   # len(getCatIds) -> 80
    #     nm = cocoGt.loadCats(catId)[0]
    #     precision = precisions[:, :, idx, 0, -1]
    #     precision = precision[precision > -1]
    #     if precision.size:
    #         ap = np.mean(precision)
    #     else:
    #         ap = float('nan')
    #     results_per_category.append((f'{nm["name"]}' ,f'{float(ap):0.3f}'))
    # print(f'\n{results_per_category}')  

if __name__ == "__main__":
    main()
