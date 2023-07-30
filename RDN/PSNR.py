import numpy as np
import cv2
import os
import sys
import time
import shutil

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

def check_shape(img0, img1):
    if not img0.shape == img1.shape:
        raise ValueError("输入的图片必须有相同的维度")
    return

def MSE(img0, img1):
    check_shape(img0, img1)
    img0 = np.asarray(img0, dtype=np.float32)
    img1 = np.asarray(img1, dtype=np.float32)
    return np.mean((img0 - img1) ** 2, dtype=np.float64)

type_dict = {float     : (-1, 1),
             np.float16: (-1, 1),
             np.float32: (-1, 1),
             np.float64: (-1, 1),
             np.float_ : (-1, 1),
             np.int8   : (-128, 127),
             np.uint8  : (0, 255)
             }

def PSNR(img_gt, img_hr, MAX=None):

    check_shape(img_gt, img_hr)

    if MAX is None:
        if img_gt.dtype.type != img_hr.dtype.type:
            raise ValueError("两张图片的数据类型需要一致")
        min_dtype, max_dtype = type_dict[img_gt.dtype.type]
        min_value, max_value = np.min(img_gt), np.max(img_gt)
        if max_value > max_dtype or min_value < min_dtype:
            raise ValueError("图片取值超出类型范围")
        MAX = max_dtype - min_dtype
    mse = MSE(img_gt, img_hr)
    c = img_gt.shape[2] # HWC
    return 10 * np.log10((MAX ** 2) / mse)# / c

def BGR_to_RGB(cvimg):
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg

def calculate_one_PSNR():
    img1 = cv2.imread(r'E:\GitLab\metrics\compile_workspace\results_onboard\RDN\rdn_quantize_img1.png')
    image_size = [512,512]
    img0 = cv2.imread(r'\\192.168.121.10\04_Model zoo\metrics\Demos\Super Resolution\RDN\section8-image.png')
    image0 = cv2.resize(img0, image_size, interpolation=cv2.INTER_CUBIC)
    image_array0 = np.array(image0,dtype='u1')
    image_array1 = np.array(img1  ,dtype='u1')
    cv2.imwrite(r'E:\GitLab\metrics\RDN\resized_image0.jpeg', image0, [int(cv2.IMWRITE_JPEG_QUALITY),95])
    print("PSNR is : " + str(PSNR(image_array1,image_array0)))

def average(arg):
    sum_ = sum(arg)
    num = len(arg)
    return sum_ / num

def collect_HR_images():
    dataset = r'E:\GitLab\metrics\datasets\Set14'
    target_path = r'E:\GitLab\metrics\datasets\Set14_original'
    for series in os.listdir(dataset):
        for each in os.listdir(os.path.join(dataset,series)):
            if "_HR" in each:
                source = os.path.join(dataset,series,each)
                target = os.path.join(target_path,series,each)
                shutil.copy(source,target)
                # print(each)
# collect_HR_images()

def calculate_one_PSNR():
    image_size = [512,512]
    img0 = cv2.imread(r'E:\GitLab\metrics\compile_workspace\results_sim\RDN_8_BY\image_SRF_2\img_002_SRF_2_ScSR.png')
    img0 = cv2.resize(img0, image_size, interpolation=cv2.INTER_CUBIC)
    img1 = cv2.imread(r'\\192.168.121.10\04_Model zoo\datasets\Set14\image_SRF_2\img_002_SRF_2_HR.png')
    img1 = cv2.resize(img1, image_size, interpolation=cv2.INTER_CUBIC)
    image_array0 = np.array(img0  ,dtype='u1')
    image_array1 = np.array(img1  ,dtype='u1')
    print(PSNR(image_array1,image_array0))
# calculate_one_PSNR()

def resize_dataset():
    dataset = r'E:\GitLab\metrics\datasets\Set14_original\image_SRF_2'
    resized = r'E:\GitLab\metrics\datasets\Set14_resized'
    image_size = [512,512]
    for img in os.listdir(dataset):
        img_name = img
        img = cv2.imread(os.path.join(dataset,img))
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_CUBIC)
        # print(img)
        # print(img_name)
        cv2.imwrite(os.path.join(resized,img_name), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
# resize_dataset()

def calculate_average_PSNR():
    log_root='./classlog'
    getLog(log_root)
    image_size = [512,512]
    PSNR_list = []
    onboard_results_path = r'E:\GitLab\metrics\compile_workspace\results_onboard\RDN_8_pc'  # 需要手动修改 上板结果路径
    origin_path = r'E:\GitLab\metrics\datasets\Set14_resized'   # 需要手动修改 原测试数据路径
    for image in os.listdir(onboard_results_path):
        img0 = cv2.imread(os.path.join(origin_path,image))
        img0 = cv2.resize(img0, image_size, interpolation=cv2.INTER_CUBIC)
        img1 = cv2.imread(os.path.join(onboard_results_path,image))
        image_array0 = np.array(img0  ,dtype='u1')
        image_array1 = np.array(img1  ,dtype='u1')
        PSNR_list.append(PSNR(image_array1,image_array0))
    # print(PSNR_list)
    avg = average(PSNR_list)
    print(avg)

calculate_average_PSNR()
