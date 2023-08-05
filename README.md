# metrics
[toc]

# CV 模型们的量化评估指标

本文档涵盖了不同 CV 任务中常用的 评估指标 (evaluation metrics)：各自的简要原理、关键公式、评估时常用的数据集、对应的代码、以及一些可能存在的注意事项。

## 图像分类 / Image Classification

### Top-1和Top-5

ImageNet有大概1000个分类，而模型预测某张图片时，会分别给出预测为这1000个类别的概率，并按概率从高到低的类别排名。用来对比的ground truth标签中，包含的是每张图中概率最高的那一个类别的序号。

Top-1 Accuracy是指预测概率排名第一的类别的与实际结果相符的准确率。Top-5 Accuracy是指预测概率排名前五的类别 包含 实际结果的准确率（注意是：预测概率排名前五中有一个是实际的结果，就认为是正确）。由此可知，Top-5的值一定高于Top-1。

## 多目标追踪 / Multi-track

主要参考文章：[Evaluation Metrics for Multiple Object Tracking | by Renu Khandelwal | Medium](https://arshren.medium.com/evaluation-metrics-for-multiple-object-tracking-7b26ef23ef5f)

MOT的模型应用场景有自动驾驶、视频监控、体育赛事中的运动员分析等等，需求侧重各不相同，对应的评价指标乍看也非常繁杂混乱，但归纳下来主要就是关注三个方面：**Detection**（检测正确性，每个框里检测到的物体是什么）、**Localization**（目标框贴合度，即物体在框里的什么位置）和**Association**（关联性，即各部分相关性）。

> We use the CLEAR metrics, including MOTA, FP, FN, IDs, etc., IDF1 and HOTA to evaluate different aspects of the tracking performance. MOTA is computed based on FP, FN and IDs. Considering the amount of FP and FN are larger than IDs, **MOTA** focuses more on the **detection performance**. **IDF1** evaluates the **identity preservation** ability and focus more on the **association performance**. **HOTA** is a very recently proposed metric which explicitly balances the **effect of performing accurate detection, association and localization**.

MOTA主要关注检测结果对象的正确性，对应Detection。IDF1主要关注的是追踪对象的细节特点，观察框内各部分的相关性，看看这个框里的手、脚、头是同一个人的，还是不同人的躯干被强行拼凑了，对应Association。HOTA是一个很新的指标，它很好地平衡了模型在Detection、Localization和Association三个方面的表现，属于非常综合的一个指标。

### MOT精度指标的特性

MOT的结果有五种类型的错误——漏检(False Negative)、多检(False Positive)、擦身而过时ID号错换(Mergers)、好端端地追踪着一个目标的ID号突然换了(Deviation)，以及追踪到一半突然不追踪了(Fragmentation)。

鉴于MOT评估可以有这么多种错误，对应的评价指标也应有单一性，各种基本的错误都应有不同的对应指标来反映。

![image-20230705170252413](C:\Users\fangshuli\AppData\Roaming\Typora\typora-user-images\image-20230705170252413.png)

**最常用的指标**

* Track-mAP
* MOTA (Multi-Object Tracking Accuracy)
* MOTP (Multi-Object Tracking Precision)
* IDF1
* HOTA (Higher-Order Tracking Accuracy)

### MOTA

MOTA一直都是和人类视觉感受最一致的一个指标。

$$
\text{MOTA} = 1- \frac{\text{FN}+\text{FP}+\text{IDSW}}{\text{GT}} \in (-\infty,1]
$$

MOTA并不关注localization的错误，detection表现的权重也远远多于association的表现。

FN 为 False Negative（漏报），整个视频漏报数量之和。
FP 为 False Positve（误报），整个视频误报数量之和。
IDSW 为 ID Switch（ID 切换总数，误配）：上图图 (a)，从红色的切换到了蓝色，记为一个 IDSW，整个视频误配数量之和，其值越小越好。
GT 是 Ground Truth 物体的数量，整个视频 GT 数量之和。
MOTA 越接近于 1 表示跟踪器性能越好，由于有跳变数的存在，当看到 MOTA 可能存在小于 0 的情况。MOTA 主要考虑的是 tracking 中所有对象匹配错误，主要是 FP、FN、IDs、MOTA 给出的是非常直观的衡量跟踪其在检测物体和保持轨迹时的性能，与目标检测精度无关。

   MOTA注重检测器的**正确性**，不能反映MOT算法对同一个目标轨迹长时间跟踪性能的好坏。

### MOTP

Multiple Object Tracking Precision，表示得到的检测框和真实标注框之间的重合程度。

$$
MOTP = \frac{\sum_{t,i}d_{t,i}}{\sum_{t}c_t}
$$

   其中$d_{t,i}$表示$t$时刻目标真实位置（ground truth）和追踪位置的距离度量；$c_t$是$t$时刻检测框和已有轨迹的匹配数。

MOTP主要量化了检测器**贴合度**的准确性，对追踪器的整体性能的体现非常之少。

MOTA和MOTP评估的都是跟踪系统的直观特征。

### IDF1

IDF1主要关注**关联性**的表现，在MOTChallenge benchmark上属于第二梯队的精度指标。

**识别 F 值 (Identification F-Score) 是指每个行人框中行人 ID 识别的 F 值。**
$$
IDF_1 = \frac{\text{2IDTP}}{\text{2IDTP}+\text{IDFP}+\text{IDFN}}
$$

IDF1：ID的F1得分，和MOTA一样是MOT中主要的评价指标。

$$
IDF_1 = \frac{2*IDTP}{2*IDTP + IDFP + IDFN}
$$

> $$
> P = \frac{TP}{TP+FP} = \frac{TP}{C} \\
> R = \frac{TP}{TP+FN} = \frac{TP}{T} \\
> F_1 = 2\frac{PR}{P+R} = \frac{TP}{\frac{T+C}{2}} = \frac{2TP}{T+C}
> $$
>



### MT&ML&IDs

Mostly Tracked，ground truth中标注的轨迹在时长上被追踪器得到的轨迹覆盖高于80%的比率。

Mostly Lost，ground truth中标注的轨迹在时长上被追踪器得到的轨迹覆盖低于20%的比率。

IDS：ID Switch，目标ID切换的次数。

### 举例说明

<img src="./Metrics.assets/mot-metrics-example.png" alt="mot-metrics-example" style="zoom:85%;" />

上图举例表示了两个追踪器对于某一目标的ID追踪情况，假设检测框都是完美匹配的，即和检测相关的指标$FN$和$FP$均为0。图中第一行表示数据集的标注情况，第二行和第三行分别为两个不同的追踪器产生的ID序列。

现记$IDTP$、$IDFP$、$IDFN$分别表示ID的正确分配次数、ID错误分配正ID次数、ID错误分配负ID次数。

对于Track 1:

1. 整个过程ID切换了3次，即$IDS=3$

   $$
   MOTA = 1 - \frac{0+0+3}{6} = 0.5
   $$
2. 追踪的起始ID为1，$IDTP=2$，$IDFP=IDFN=4$

$$
IDF1 = \frac{2*2}{2*2+4+4} = 0.333
$$

对于Track2:

1. 整个过程ID切换了3次，即$IDS=3$

   $$
   MOTA = 1 - \frac{0+0+3}{6} = 0.5
   $$
2. 追踪的初始ID为1，$IDTP=4$，$IDFP=IDFN=2$

   $$
   IDF1 = IDF1 = \frac{2*4}{2*4+2+2} = 0.667
   $$

## 目标追踪

### OPE：Precision Plot & Success Plot

#### Precision Plot

> Percentages of the frames whose estimated locations lie in a given threshold distance to ground-truth centers.

追踪算法估计的目标位置（用bounding box表示）的中心点和目标ground-truth的中心点的距离小于给定阈值的视频帧的百分比。不同的阈值可以得到不同的百分比，通常设定阈值为20个像素点。

例子：视频有101帧，阈值设为20个像素点，经过追踪算法后得到bounding box和ground truth中心点距离小于阈值的有60帧，其余40帧的中心点距离都大于20个像素点。因此，当阈值为20个像素点时，精度为0.6。

缺点：无法反映目标物体大小和尺度的变化。

#### Success Plot

> Let rt denote the area of tracked bouding box and ra denote the ground truth. An Overlap Score (OS) can be defined by $S=\frac{|rt \cap ra|}{|rt\cup ra|}$, where $\cap$ and $\cup$ are the intersection and union of two regions, and $|\cdot|$ counts the number of pixels in the corresponding area. Afterwards, a frame whose OS is larger than a threshold is termed as a successful frame, and the ratios of successful frames at the thresholds ranged from 0 to 1 are plotted in success plot.

![PrecisionPlot&SuccessPlot](./Metrics.assets/Precision&SuccessPlot.png)

以上两种常用的评估方式一般都是用ground truth中目标的位置初始化第一帧，然后运行根据算法，得到平均精度和成功率。这种方法称为One-Pass Evalution (OPE)。OPE有两个缺点：1. 跟踪算法可能对第一帧给定的初始位置敏感，不同的帧初始化会造成较大的影响；2. 大多数算法遇到跟踪失败后没有重新初始化的机制。

### 鲁棒性评估

通过从时间和空间上打乱，再进行评估。时间上打乱指的是从不同的起始帧开始评估；空间上打乱指的是使用不同的bounding box进行评估。因此，可以分为：Temporal Robustness Evaluation (TRE) 和 Spatial Robustness Evaluation (SRE)。

#### Temporal Robustness Evaluation

> Each tracking algorithm is evaluated numerous times from different starting frames across an image sequence. In each test, an algorithm is evaluated from a particular starting frame, with the initialization of the corresponding ground truth object state, until the end of an image sequence. The tracking results of all the tests are averaged to generate the TRE score.

例如，从测试视频的第一帧、第十帧、第二十帧开始跟踪，每次初始化的bounding box就是对应帧的ground truth。最后，对这些帧的结果取平均值，得到TRE score。

#### Spatial Robustness Evaluation

> To evaluate whether a tracking method is sensitive to initialization errors, we generate the object states by slightly shifting or scaling the ground-truth bounding box of a target object. In this work, we use eight spatial shifts (four center shifts and four corner shifts), and four scale variations. The amount for shift is 10 percent of the target size, and the scale ratio varies from 80 to 120 percent of the ground truth at the increment of 10 percent. The SRE score is the average of these 12 evaluations.

需要注意的是：有些算法对于初始化给定的bouding box敏感，初始化的bounding box是人工标注的。自然地，评估网络对于初始化bounding box的鲁棒性的做法就是“适当地”将ground truth轻微的平移以及尺度的扩大和缩小。“适当地”具体表现为：平移的大小取目标物体大小的10%；尺度变化范围为ground truth的80%到120%，间隔10%。最后，取这些结果的平均值作为SRE score。

![SRE&TRE](./Metrics.assets/SRE&TRE.png)

## 目标检测 / Object Detection

在目标检测领域最重要的 metric 是 AP (Average Precision)，而根据IoU (Intersection over Union) 阈值的不同分为$\text{AP}_{50}$、$\text{AP}_{75}$等。除了AP之外，还有AR (Average Recall)，这个在下文会具体解释。

<img src="./Metrics.assets/Actual-Predicted.png" alt="Actual vs Predicted: TP, FP, FN, & TP" style="zoom: 67%;" />

### AP (Average Precision)

精确度Precision的计算公式为：

$$
\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}}
$$

- TP (True Positive)，真阳性，即正确的检测结果；
- FP (False Positive)，假阳性，就是检测错了；
- FN (False Negative)，假阴性，即漏检，也就是本来应该有框但没有检测出来。

上表中的**Positive/Negtive**表示**检测出来/没有检测出来**，Precision 测量的是 **所有检测出来的结果中检测正确的比率**，

那问题来了，怎么才算检测出来了呢？通常在检测网络中，会定义IoU这个值来作为判断“是否检测出目标”的阈值。例如当IoU取值0.5时，表示检测框和ground-truth的交并比大于0.5时才算做“检测出目标”，即Positive，否则为Negtive。

### mAP (mean Average Precision)

COCO的方法中，IoU取值$[0.5:0.05:0.95]$这10个点，然后对这10个点的AP进行平均得到mAP。

```python
self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
```

在 [COCO 官网](https://cocodataset.org/#detection-eval)上，有如下的说明：

> Unless otherwise specified, *AP and AR are averaged over multiple Intersection over Union (IoU) values*. Specifically we use 10 IoU thresholds of .50:.05:.95. This is a break  from tradition, where AP is computed at a single IoU of .50 (which corresponds to our metric APIoU=.50). Averaging over IoUs rewards detectors with better localization.

### AR (Average Recall)

$$
\text{AR} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

Recall 测量的是**所有实际、应该有的目标中正确检出目标的比率**。

### AP 和 AR 的关系

在实际中，取决于对检测模型输出的 score 的阈值，我们需要对 Precision 与 Recall 进行权衡取舍。过高的 score 阈值会提高 Precision 但降低 Recall，反之亦然。而通过微调 score 阈值，我们能够得到许多对 `(Precision, Recall)` 的数据。将它们作为二维坐标画在坐标系中，我们一般能得到类似于如下形状的曲线。

<img src="Metrics.assets/precision-over-recall.jpg" alt="Precision over Recall" style="zoom: 33%;" />

那么，Average Precision (AP) 即是这一 Precision-Recall 曲线的 **曲线下面积 (Area under Curve)**。具体的计算过程可以参照 [这篇博文](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173) (英语，需翻墙) 中的例子。

### COCO val 数据集对模型进行评测

[COCO 的官方文档](https://cocodataset.org/#detection-eval)

对应的代码库位于 GitHub  上的 [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)。

此外，我还写了一个笔记本 yolo2coco.ipynb（位于 Object Detection 目录下），将 YOLO 模型输出的 txt 转换为 COCO 所需的 json，并调用 cocoEval 对我们的 YOLOv5 模型的输出进行评估。

#### 数据准备

在调用脚本进行评估之前，我们需要下载 COCO 2014 或 2017 的 val set 及其对应的 annotations。注意：COCO test 数据集的 annotation 是不对外公开的，无法用于我们的评估。而 train 数据集被用于训练，最好不要用同一数据集进行评估（避免过拟合导致的结果不准确）。

随后，我们应当运用模型对 val set 中的图片进行推理，并将结果输出为一个 json 文件。json 文件格式如下，由一个大的 list 组成，list 中包含了 每个检测结果：图片 ID，分类，检测框坐标，score。

```json
[
    {
        "image_id": 42,
        "category_id": 18,
        "bbox": [
            258.15,
            41.29,
            348.26,
            243.78
        ],
        "score": 0.236
    }
]
```

#### 运行评估

随后，我们可以将这个 json 文件与 ground truth 的 annotations 传入 `COCOEval` 类中并进行评估。

```python
# annFile 为 ground truth json 的路径
cocoGt=COCO(annFile)
# resFile 为检测结果 json 的路径
cocoDt=cocoGt.loadRes(resFile)

cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
```

脚本会输出如下的输出，包含了 COCO 所有的 metrics： AP, AP-50, AP-75, AP-s, AP-m, AP-l 以及相对应的 AR。不过一般我们只使用各 AP 的值对模型进行衡量。

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.505
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.697
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.573
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.586
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.387
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.594
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.595
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.564
```

## 旋转框检测 / Rotation Detection

> We use four vertex coordinates to represent the Orientation Bounding Box positions, where (x1,y1) is the starting point and the other points are labeled as (x2,y2), (x3,y3), and (x4,y4) in a **clockwise** direction in order with it as the starting point. We find that the Orientation Bounding Box used to display the detected GCP(ground control  points)-markers starts mainly at the bottom-right, so we can use Equation (1) to calculate the position of the GCP.

### 旋转坐标定位方法

$$
\left\{
\begin{aligned}
[][x,y]^T & = & [(j-1)*w,(i-1)*h]^T+[x_1，y_1]^T,right down\\
[x,y]^T & = & [(j-1)*w,(i-1)*h]^T+[x_2，y_2]^T, left down\\
[x,y]^T & = & [(j-1)*w,(i-1)*h]^T+[x_3，y_3]^T, left up\\
[x,y]^T & = & [(j-1)*w,(i-1)*h]^T+[x_4，y_4]^T, right up\\
[x,y]^T & = & [(j-1)*w,(i-1)*h]^T+[\Sigma x_i，\Sigma y_i]^T/4, cross\\
\end{aligned}
\right.\tag{1}
$$

> where w, h, are the width and height of the detected image; i, j, are the number of rows of the cropped image in the original image; BR, BL, TR, TL, and CR refer to bottom-right, bottom-left, top-right, top-left and cross-shaped GCP-markers types, respectively.

#### 数据准备

旋转框标签含义如下：

    "bbox": [
                587.0,	右下
                456.0,	左下
                27.0,	左上
                22.0	右上
            ],

在使用官方cocoAPI进行精度测试之前，需要将芯片推理的四个点坐标（obb格式）转换为hbb格式，即左上角坐标 x, y, 和 w, h。接下来的流程和coco测精度一致。obb转换为hbb：{ class_id, point_x1, point_y1, point_x2, point_y2, point_x3,point_y3,point_x4,point_y4,score } -> {class_id, x, y, w, h, score}

注意事项：标注结果也需要进行obb转换为hbb。

## 显著性检测 / Saliency Detection

**注**：我们模型库中的 $U^2$ Net 其实是一种显著性检测模型，而不是图像分割模型。

与[图像分割](#分割--segmentation)（标注所有目标的类别与轮廓）相比，显著性检测仅会标注图像中最显著（人类观测者最感兴趣）的物体轮廓。在传统方法中，显著性检测倾向通过颜色、对比等特征提取寻找显著性物体，而语义分割倾向使用边缘检测。但在深度学习的方法中，两者的网络结构与 loss 函数都是相接近的。一些常见的显著性检测的应用是区分照片中的前景与背景。

<img src="Metrics.assets/u2netqual.png" alt="Input & Output of Saliency Detection Model" style="zoom: 50%;" />

以 $U^2$ Net 模型为例，其输入与输出如上图。模型的输出是一种被称为 Saliency Map (显著图) 的 mask。对显著性检测模型进行的评估将围绕其 Saliency Map 与数据集给出的 Ground Truth 的比较展开。

最常用的 metrics 有 MSE 与各类 F-分数 ($F_\beta, F_\beta^w, \dots$)。

### 显著性检测中的 Precision & Recall

回忆目标检测中有关 [AP 的基本原理](#average-precision-ap-的基本原理) 节中与 Precision 和 Recall 有关的段落。在显著性检测中，Precision & Recall 被相似地定义为：

$$
\text{Precision} = \frac{|M \cap G|}{|M|}
$$

$$
\text{Recall} = \frac{|M \cap G|}{|G|}
$$

其中，$M$ 为模型输出的 Salient Map，$G$ 为数据集标注的 Ground Truth。绝对值为面积。

与目标检测相似，显著性检测同样可以使用 Precision-Recall 曲线对模型进行评估，曲线下面积 (Area Under Curve) 越大，模型越优秀。但显著性检测并不同目标检测一样将 AP 作为一种 metric。

### F-分数 / F-measure

#### F-beta $F_\beta$

尽管 $F_1$  是最经典的 F-分数，在显著性检测中最常用的为 $F_\beta$

$$
F_\beta = (1+\beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{(\beta^2 \cdot \text{precision}) + \text{recall}}
$$

其中，一般 $\beta^2 = 0.3$，以增加 Precision 的权重。这一做法最早出现于论文 [Frequency-tuned salient region detection (CVPR 2009)](https://ieeexplore.ieee.org/document/5206596) 中，并被沿用下来。

**注**：在 $U^2$ Net 中，作者仅仅报告了数据集中最大的（而非平均） $F_\beta$ 值，并将其命名为 $maxF_\beta$。

#### 其它

此外，还有一些引申的其它 F-分数，但由于使用较少所以不再展开。

- Weighted F-measure $F_\beta^w$
  - 提出于论文 [How to Evaluate Foreground Maps (CVPR 2014)](https://ieeexplore.ieee.org/abstract/document/6909433) 中
- Relax boundary F-measure $relaxF_\beta^b$
  - 提出于论文 [Precision and Recall for Ontology Matching](https://hal.inria.fr/hal-00922279/) 中

### 平均绝对误差 / MAE

MAE (Mean Absolute Error) 计算的是 Saliency Map 与 Ground Truth 之间每像素的平均差异。

$$
MAE = \frac{1}{W\times H} \sum^{W}_{x=1} \sum^{H}_{y=1} \left| M(x,y)-G(x,y) \right|
$$

其中 $M(x,y)$ 指代 Saliency Map 在位置 $(x,y)$ 处的值，$G(x,y)$ 为 ground truth 在位置 $(x,y)$ 处的值。

### 运用 DUTS-TE 数据集进行评估

我基于 [Mehrdad-Noori/Saliency-Evaluation-Toolbox](https://github.com/Mehrdad-Noori/Saliency-Evaluation-Toolbox) 中的代码，做了一个笔记本 `Saliency Evaluation.ipynb` 放在了 Saliency Detection 的目录下。这个笔记本可以在给定 Saliency Map 与 Ground Truth 输出图片路径的情况下，同时计算多个 metrics。

```python
# 模型输出的 Saliency Map 的结果
sm_dir = R'D:\Ye Shu\Evaluation Metrics\Saliency Detection\U-2-Net\test_data\u2netpDUTS_results'
# 数据集给出的 Ground Truth
gt_dir = R'D:\Ye Shu\Evaluation Metrics\Saliency Detection\DUTS-TE\DUTS-TE-Mask'

# 进行计算，第三个参数可以有选择性地传入需要计算的 metrics，降低运行时间。
res = calculate_measures(gt_dir, sm_dir, ['MAE', 'E-measure', 'S-measure', 'Max-F', 'Adp-F', 'Wgt-F'], save=False)
```

## 分割 / Segmentation

关于分割与显著性检测的区别，请参照[上节](#显著性检测--saliency-detection)。

对于分割，又可细分为语义分割 (semantic segmentation)、实例分割 (instance segmentation)、与全景分割 (panoptic segmentation)。

![Difference between Semantic, Instance, & Panoptic Segmentation](Metrics.assets/semantic-vs-instance-panoptic-segmentation.jpg)

其中，`<u>`语义分割 `</u>` 的工作是对图像中的所有像素标注为其所属目标的类型；最终的输出中，属于同一目标类型的像素会被聚类到一起（如上图 b 中不同车辆显示的是同一颜色）。

`<u>`实例分割 `</u>` 的工作基于目标检测之上，会识别出图像中每一个独特的目标实例（例如车辆）并对检测框中的目标进行分割。其特点是同一类型的不同目标会被区分开来（如上图 c 中不同车辆由不同颜色进行表示）。

`<u>`全景分割 `</u>` 其实是对语义分割与实例分割的一个融合。在标注每个像素为所属目标类别的同时对不同的目标实例进行有区分地分割。在一些论文中，也有人使用 “实例分割” 来指代全景分割，尽管实例分割并不保证为所有像素进行类别标注。

### 语义分割 / Semantic Segmentation

语义分割也常被称作 **图像分割 (Image Segmentation)**。通常使用 Pixel Accuracy 与 mean IoU 进行评估。常用的数据集为 CityScapes（[官网](https://www.cityscapes-dataset.com/) 需要账号） 与 PASCAL VOC （[牛津官网](https://host.robots.ox.ac.uk/pascal/VOC/) 疑似 down 了，[第三方镜像](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)）。此外，COCO Stuff （[官网](https://cocodataset.org/#stuff-eval)） 也可以被用作评估，但被使用的频率要远低于 CityScapes。

#### Pixel Accuracy

$$
\text{Pixel Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}
$$

|            | 标注为 X            | 标注为 非X          |
| ---------- | ------------------- | ------------------- |
| 实际为 X   | TP (True Positive)  | FN (False Negative) |
| 实际为 非X | FP (False Positive) | TN (True Negative)  |

这一计算方法的问题是，对于图片中 X 类别占比较小的，TP 值会远远小于 TN；那么即使模型将整张图片标注为 非X ，也会取得较高的 Pixel Accuracy。

#### Dice

<img src=".\Metrics.assets\dice.png" style="zoom:67%;" />

上图中，ground truth用蓝色标记，网络输出的预测结果用绿色标记，两者的交集是红色部分。显然，红色部分就是true positive，即这些像素本来就被标记成类别A，如今也确实被检测成了类别A。其他的两部分分别表示false positive（剩下的绿色）和false negative（剩下的蓝色）。因此Dice的最大值为1， 最小值为0。则Dice可以表示为：

$$
\text{Dice} &= \frac{2N_{\text{true positive}}}{2N_{\text{true positive}} + N_{\text{flase positive}} + N_{\text{false negative}}} \\
&=\frac{\left| X \cap Y \right|}{|X| + |Y|}
$$

其中$X$和$Y$分别表示ground truth和检测结果的像素集合。

#### mIoU

此外，我们也可以通过 IoU (Intersection over Union) 对模型进行评估。IoU又称为Jaccard指数，所以在一些指标中会用到Jaccard，表示的就是IoU。

$$
\text{IoU} &= \frac{\text{Intersection}}{\text{Union}} \\ 
&= \frac{|X \cap Y|}{|X \cup Y|}\\
&= \frac{|X \cap Y|}{|X| + |Y| - |X \cap Y|} \\
& = \frac{TP}{TP+FP+FN}
$$

mean IoU (mIoU) 则是对全部类别分别计算 IoU 并对其取平均值求得。

设共有$k+1$个类，为$L_0,\dots,L_k$，其中包含一个空类或背景类。$p_{ij}$表示属于类 $i$ 但被预测为类 $j$ 的像素数量。当$i=j$ 时， $p_{ii}$ 表示 True Positive 的像素数量；$p_{ij}$ 表示 False Positive 的像素数量；$p_{ji}$ 表示 False Negative 的像素数量。

于是，mIoU表示为：

$$
\text{mIoU} = \frac{1}{k+1} \sum\limits_{i=0}^{k} \frac{p_{ii}}{\sum\limits_{j=0}^{k}p_{ij} + \sum\limits_{j=0}^{k}p_{ji}- p_{ii}}
$$

#### Frequency Weighted IoU

在mIoU的基础上，通过每个类出现的频率来设置其权重：

$$
\text{FWIoU} = \frac{1}{\sum\limits_{i=0}^{k}\sum\limits_{j=0}^{k}p_{ij}} \sum\limits_{i=0}^{k} \frac{p_{ii}}{\sum\limits_{j=0}^{k}p_{ij} + \sum\limits_{j=0}^{k}p_{ji} - p_{ii}}
$$

#### 计算实例

运用 COCO 进行评估的代码可以见 [COCOStuffAPI](https://github.com/nightrome/cocostuffapi) 代码库中的 [cocoStuffDemo.py](https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/cocostuff/cocoStuffDemo.py)。其使用方法与 COCO Detection 的评估（见 [对应章节](#运用-coco-val-数据集对模型进行评测)）是非常相似的。

### 实例分割 / Instance Segmentation

在对实例分割模型的评估中，最常用的数据集为 COCO。CityScapes 的使用频率要远小于 COCO。

其评估方法与 Metrics 完全同 目标检测，以 AP 作为最终 metrics。详见 [目标检测章节](#目标检测--object-detection)。唯一区别是，计算 IoU 时，计算的不再是检测框的 IoU，而是 segmentation mask 的 IoU。

在 [COCO 官网](https://cocodataset.org/#detection-eval) 上，有如下的说明：

> The evaluation metrics for detection with bounding boxes and segmentation masks are identical in all respects except for the IoU computation (which is performed over boxes or masks, respectively).


## 姿态估计 / Pose Estimation

姿态估计可以细分为四个任务：

- 单人姿态估计 (Single-Person Skeleton Estimation)
- 多人姿态估计 (Multi-Person Pose Estimation)
- 人体姿态跟踪 (Video Pose Estimation)
- 3D人体姿态估计 (3D Skeleton Estimation)

不同的数据集有不同的评价指标。

### OKS

> The COCO evaluation defines the object keypoint similarity (OKS) and uses the mean average precision (AP) over 10 OKS thresholds as main competition metric. The OKS plays the same role as the IoU in object detection.

COCO给的指标。基于目标检测中IoU思想，COCO对于姿态识别也同样提出了OKS (Object Keypoint Similarity)。

> We do so by defining an object keypoint similarity (OKS) which plays the same role as the IoU.

计算公式：

$$
\text{OKS}_p = \frac{\sum\limits_{i=1}^{N}e^{-\frac{d_{pi}^2}{2s_p^2k_i^2}}\delta(v_i>0)}{\sum\limits_{i=1}^{N}\delta(v_i>0)}
$$

- $d_{pi}$ ：第$p$个人的第$i$个预测的关键点和ground truth之间的欧氏距离；
- $s_p$ ：即第$p$个人的scale，表示这个人所占面积大小的平方根，通过ground truth的annotation中的box计算得到，$s = \sqrt{(x_2-x_1)(y_2-y_1)}$ ；
- $k_i$ ：骨骼点的归一化因子，这个因子通过对已有数据集中所有ground truth计算的标准差而得到，$\sigma^2=\text{E}[d_i^2/s^2]$，一般取$k_i = 2*\sigma_i$。

一般来说，都只会关注那些明确标注了的关键点，即对$\forall i=1,2,\dots,N，$有$v_i=1$，即$\delta(v_i>0) =1$。

$$
\text{OKS}_p = \frac{1}{N}\sum\limits_{i=1}^{N}e^{-\frac{d_{pi}^2}{2s_p^2k_i^2}}
$$

**OKS矩阵**

目标检测肯定会存在漏检、多检测的情况（相比于ground truth），例如网络检测到$P$个人，而ground truth中标注有$Q$个人，则应当建立$P\times Q$的矩阵：

$$
\begin{equation}
\begin{bmatrix}
\text{OKS}_{1,1} & \text{OKS}_{1,2} & \cdots & \text{OKS}_{1,Q} \\
\text{OKS}_{2,1} & \text{OKS}_{2,2} & \cdots & \text{OKS}_{2,Q} \\
\vdots           & \vdots           & \ddots & \vdots           \\
\text{OKS}_{P,1} & \text{OKS}_{P,2} & \cdots & \text{OKS}_{P,Q}  
\end{bmatrix}
\end{equation}
$$

每一行中会有一个最大值，对应的是检测到的目标数据和ground truth最接近的那一个OKS。

对于检测到的数据，存放的是关键点的坐标：

$$
\boldsymbol{d} &= [x_1,y_1,v_1,\dots, x_K,y_K,v_K]\\ 
&= [x_1,y_1,1,\dots,x_K,y_K,1]
$$

**数据格式**

在COCO数据集的[数据格式]([COCO - Common Objects in Context (cocodataset.org)](https://cocodataset.org/#format-data))里罗列了对应各种任务的数据集的annotations JSON的格式。对于关键点检测，在目标检测annotation的基础上，增加了关键点的标注：

```json
annotation{
	"keypoints": [x1,y1,v1,...], "num_keypoints": int, "[cloned]": ...,
}

categories[{
	"keypoints": [str], "skeleton": [edge], "[cloned]": ...,
}]
```

对于网络的输出结果，也需要按照指定格式来：

```json
[{
	"image_id": int, 
    "category_id": int, 
    "keypoints": [x1,y1,v1,...,xk,yk,vk], 
    "score": float,
}]
```

网络输出结果的json文件中存放的是列表，元素只有一个，为四个属性的键值对。

**代码**

我的格式转换程序如下：

```python
import json
import os 
import cv2
import re
import numpy as np

# 测试图片和icraft输出结果路径（每张图片保存成对应的txt文件）
img_root = r"../eval_img"
result_root = r"../result_from_icraft"       # txt
img_name = os.listdir(img_root)
kpoints = 18

dataset = []   # dataset用来保存json，COCO的格式就是一个数组，里面放字典

pattern = re.compile('^0+') # 从字符串开头开始匹配一个或多个0
for i in range(len(img_name)):
    txt_name = img_name[i].split(".")[0] + ".txt" 
    # 如果图片没有做测试，就不计入json中
    if not os.path.exists(os.path.join(result_root, txt_name)): 
        continue
    img_id = re.sub(pattern, '', img_name[i]).split('.')[0]  # 把图片名字中前面的0全去掉，这样id就和COCO给的id一致

    with open(os.path.join(result_root, txt_name), "r") as f:
        result_data = f.readlines()  # 读入数据，每张图都会对应18*n的行数，其中n为检测到的人数
        person_num = int(len(result_data) / 18)
  
        for i in range(person_num):
            dataset.append({"image_id": int(img_id), "category_id": 1, "keypoints":[], "score": 1})
            data_per_person = result_data[i*18 : (i+1)*18]
            for j in range(len(data_per_person)):
                if j==1:
                    continue
                tmp = data_per_person[j].strip("\n").split(" ")
                x = tmp[0]; y=tmp[1]; v=tmp[2]; scroe=tmp[3]
                dataset[-1]["keypoints"].append(float(x))
                dataset[-1]["keypoints"].append(float(y))
                dataset[-1]["keypoints"].append(float(v))

# 把上面的数据写入json文件
json_file = r"./result_json/result.json"
with open(json_file, 'w') as f:
    json.dump(dataset, f)
```

COCO的计算OKS的代码：

```python
def computeOks(self, imgId, catId):
    p = self.params
    # dimention here should be Nxm
    gts = self._gts[imgId, catId]
    dts = self._dts[imgId, catId]
    inds = np.argsort([-d['score'] for d in dts], kind='mergesort') # 根据score从大到小排序，返回的是索引
    dts = [dts[i] for i in inds]
    if len(dts) > p.maxDets[-1]: 
        dts = dts[0:p.maxDets[-1]] # maxDets是按照置信度排序后取前多少个目标的个数，默认了20
    # if len(gts) == 0 and len(dts) == 0:
    if len(gts) == 0 or len(dts) == 0:
        return []
    ious = np.zeros((len(dts), len(gts)))
    sigmas = p.kpt_oks_sigmas
    vars = (sigmas * 2)**2
    k = len(sigmas)
    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gt['keypoints'])
        xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
        k1 = np.count_nonzero(vg > 0)
        bb = gt['bbox']
        x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
        y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
        for i, dt in enumerate(dts):
            d = np.array(dt['keypoints'])
            xd = d[0::3]; yd = d[1::3]
            if k1>0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros((k))
                dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
            e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
            if k1 > 0:
                e=e[vg > 0]
            ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
    return ious
```

**关键点数量匹配**

COCO中的关键点个数是17，而网络中的关键点不一定是17，例如Openpose中的关键点个数是18，因此就需要将这些**关键点对齐**。具体地，对于COCO数据集定义的关键点（上）和OpenPose定义的关键点（下）：

<img src=".\Metrics.assets\COCOkeypoints.png" style="zoom:80%;" />

<img src=".\Metrics.assets\OpenPosekeypoints.png" style="zoom:80%;" />

### PCK

Percentage of Correct Keypoints，为检测得到的关键点与其对应的ground truth间的**归一化距离**小于设定阈值的比例。

不同数据集的归一化方式不同：

* FLIC中以躯干直径作为归一化距离
* MPII中以头部长度（两个耳朵的距离）作为归一化参考距离，即PCKh

$$
\text{PCK}^p_{\sigma} = \frac{1}{|\mathcal{D}|}\sum\limits_{\mathcal{D}}\delta(||x^p_d - x^p_g||_2 < \sigma)
$$

> In this paper, we adopted a threshold of half the distance of the ground truth limb, commonly referred as PCP@0.5. The PCP method introduces a bias due to the stronger penalization for the detection of smaller limbs (i.e. arm in comparison to torso), since they naturally have a shorter distance, and consequently a smaller threshold for detection

本文采用了PCP@0.5，也就是groundtruth肢体距离的一半。这种方法会在局部精细地区有更小的阈值。

> Another metric used is the Percentage of Correct Keypoints (PCK). This metric considers the prediction of a keypoint correct when a joint detection lies within a certain threshold distance of the ground truth. Two commonly used thresholds were adopted. The first is PCK@0.2, which refers to a threshold of 20% of the torso diameter, and the second is PCKh@0.5, which refers to a threshold of 50% of the head diameter.

PCK也是一个很常用的姿态评估指标。关键点与groundtruth的距离在一定范围内则判定为正确。本文使用了PCK@0.2和PCKh@0.5，代表阈值分别为躯干直径的20%，以及头围的50%。

其中$\mathcal{D}$表示测试集；$\sigma$表示归一化参考距离；$x_d^p$和$x_g^p$表示的是第$p$个人的检测关键点和ground truth关键点。一般来说，例如以人的头的尺寸为归一化参考距离，可以取头的0.1、0.2、0.3等来得到不同阈值下的PCK。通常会看到PCKh0.5，表示的是以0.5倍头部直径为归一化距离得到的PCK值。

## 热图法关键点检测 / Heatmap Pose


#### 后处理原理介绍

网络输出为大小(N×16×46×46)的特征图，分别对应16个关节检测点。后处理算法主要包括两类：关节点热力图生成与姿势识别图绘制。

关节点热力图生成相关算法如下：首先将输出特征图插值恢复至原图大小，对图像元素进行负值赋0的操作后再通过OpenCV的热力值转换函数，将输出转换为16张可视化的关节检测热力图并绘制；

姿势识别图绘制相关算法如下：首先将输出特征图插值恢复至原图大小，对每个通道中图像元素进行最大值索引，找出最有可能为对应关节点的图像位置，进行关键点绘制，再根据对应关节点的连接关系，进行姿势绘制。



## 图像超分辨 / Image Super Resolution

对于图像超分辨的模型，评估的流程一般为

1. 准备一张 高分辨率 (High Resolution, HR) 的原图 & 一张 低分辨率 (Low Resolution, LR) 的图（一般会同时在数据集中给出，HR 与 LR 的关系一般为 2x 和/或 4x）
2. 将 LR 图片输入模型中，得出放大（2x 和/或 4x）后的图片
3. 将放大后的图片与 HR 原图进行比对

注意：放大后的图片与 HR 的分辨率 **必须** 一致。即，如果模型放大倍数为 2x，那么应该与 2x 倍数的 HR 进行比对；如果模型放大倍数为 4x，应与 4x 的 HR 进行比对。

比对时最常用的 metrics 为 PSNR 与 SSIM，其中 PSNR 又发展自 MSE（*MSE 现在已经很少被使用*）。我们将在接下来分别展开。

此外，还有一些近些年来学界提出的其它更加优秀的 metrics。如 LPIPS ([主页](https://richzhang.github.io/PerceptualSimilarity/) | [CVPR 2018](https://arxiv.org/abs/1801.03924) | [GitHub](https://github.com/richzhang/PerceptualSimilarity)) ，以及用于评估 GAN 生成图片的 Fréchet inception distance 等，限于篇幅不再展开。

如果想要深入了解评估 GAN 模型的话，可以阅读这篇论文 [Pros and Cons of GAN Evaluation Measures](https://arxiv.org/abs/1802.03446)。

### 峰值信噪比 / PSNR

PSNR (Peak Signal-to-Noise Ratio，峰值信噪比) 是一个由 MSE 衍生出的概念。

#### 均方误差 / MSE

MSE (Mean Squared Error，均方误差) 是最简单的一种 metric。

假设我们有一张 $m \times n$ 大小的黑白（单个 channel）HR 原图 $I$，以及模型放大后同样为 $m \times n$ 大小的预测图 $K$，那么 MSE 可以通过以下公式取得：

$$
\text{MSE} = \frac{1}{mn} \sum_{i=0}^{m-1}\sum_{j=0}^{n-1}
\left( I\left(i,j\right) - K\left(i,j\right) \right)^2
$$

同样的公式对于多个 channel 的图像也同样适用，对各 channel 各像素的平方误差求和后取其均值。

在 Python 中，可以简化为如下的代码（`image0` 与 `image1` 为 `np.ndarray` 类型）：

```python
mse = np.mean((image0 - image1) ** 2, dtype=np.float64)
```

但是，因为 MSE 没有考虑到不同编码方式的峰值 (peak intensity) 不同，导致 **MSE 值并不具有统一的含义**。PSNR 解决了这一问题。

#### 峰值信噪比 / PSNR

PSNR (Peak Signal-to-Noise Ratio，峰值信噪比) 是一个由 MSE 衍生出的概念。

$$
\text{PSNR} = 10 \cdot \log_{10}\left( \frac{\text{MAX}^2}{\text{MSE}} \right)
$$

其中，$\text{MAX}$ 代表了每个数据点的最大值。对于最常见的以 8 bit 整数存储的图像而言为 255。

同样，我们可以用一行 Python 代码来计算它。

```python
psnr = 10 * np.log10((data_range ** 2) / mse)
```

#### 数值意义

> 图像与影像压缩中典型的峰值信噪比值在 30dB 到 50dB 之间，愈高愈好。
>
> - PSNR接近 50dB ，代表压缩后的图像仅有些许非常小的误差。
> - PSNR大于 30dB ，人眼很难察觉压缩后和原始影像的差异。
> - PSNR介于 20dB 到 30dB 之间，人眼就可以察觉出图像的差异。
> - PSNR介于 10dB 到 20dB 之间，人眼还是可以用肉眼看出这个图像原始的结构，且直观上会判断两张图像不存在很大的差异。
> - PSNR低于 10dB，人类很难用肉眼去判断两个图像是否为相同，一个图像是否为另一个图像的压缩结果。
>
> 摘自 [维基百科](https://zh.wikipedia.org/wiki/%E5%B3%B0%E5%80%BC%E4%BF%A1%E5%99%AA%E6%AF%94#%E6%95%B8%E5%80%BC%E4%BB%A3%E8%A1%A8%E7%9A%84%E6%84%8F%E7%BE%A9) （需翻墙）

> Typical values for the PSNR in lossy image and video compression are between 30 and 50 dB, provided the bit width is 8 bits, where higher is better.
>
> The processing quality of 12-bit images is considered high when the PSNR value is 60 dB or higher.
>
> For 16-bit data typical values for the PSNR are between 60 and 80 dB.
>
> Acceptable values for wireless transmission quality loss are considered to be about 20 dB to 25 dB.
>
> from [Wikipedia](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)

#### 缺点 & 为什么使用 SSIM

然而，PSNR 与 MSE 的问题在于，其计算的是像素点之间“绝对”的数值错误。而同样数值的绝对错误对于观者而言则可能会由非常不同的效果（见下图，误差图像的 MSE 为 144，PSNR 为 26.54）

例如，几个像素点的大规模偏差（图4）与多个像素点的小规模偏差（图5）可能导致相同的绝对误差（MSE & PSNR），但两张图却会有完全不同的观感。

![Same MSE Value can produce images with very different looks](Metrics.assets/ssim-mse.png)

为了解决这个问题，NYU 与 UT Austin 的研究者们提出了 SSIM，以试图计算图像之间“结构化”的区别，进而模拟两张图对于人类观众的整体不同程度。这一评估方法对于超分辨来说显然比起 PSNR 要更为有效一些。

### 结构相似性 / SSIM

详见论文 [Image quality assessment: from error visibility to structural similarity](https://ieeexplore.ieee.org/abstract/document/1284395) ([PDF](https://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf))

#### SSIM 的结构

SSIM 会从图片中针对三个不同维度的特征进行比较：亮度 (luminance, $l$)、对比度 (contrast, $c$)、与结构 (structure, $s$)。

![Structure of SSIM](Metrics.assets/ssim.png)

其整个计算过程可以分为 测量 (Measurement), 比较 (Comparison), 合并 (Combination) 三个步骤

#### 测量 (measurement) 函数

其中，亮度 (luminance) 的测量方式为取各像素的平均

$$
\mu_x = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

对比度 (contrast) 的测量方式是取各像素的 标准差 (standard deviation)

$$
\sigma_x = \left( \frac{1}{N-1} \sum_{i=1}^{N} (x_i-\mu_x)^2 \right)^{\frac{1}{2}}
$$

#### 比较 (comparison) 函数

对两个输入的 亮度 的比较则会比较双方的 $\mu$。其中 $c_1 = (K_1L)^2$ 为常数：$K_1$ 为任意常数，$L$ 为每个数据点的范围（一般为 255）

$$
l(x,y) = \frac{2\mu_x\mu_y + c_1}{\mu_x^2 + \mu_y^2+c_1}
$$

对两个输入的 对比度 的比较则会比较双方的 $\sigma$，其中 $c_2 = (K_2L)^2$ 为常数（$K_2$ 为任意常数）

$$
c(x,y) = \frac{2\sigma_x\sigma_y+c_2}{\sigma_x^2+\sigma_y^2+c_2}
$$

对于 结构 (structure) 的比较，我们会将输入信号的平均除以标准差。其中一般 $c3=\frac{c2}{2}$。

$$
s(x,y) = \frac{\sigma_{xy}+c_3}{\sigma_x\sigma_y+c_3}
$$

其中

$$
\sigma_{xy} = \frac{1}{N-1} \sum_{i=1}^{N} (x_i-\mu_x)(y_i-\mu_y)
$$

#### 合并 (Combination)

将三个比较方式进行加权（$\alpha, \beta, \gamma$）后相乘，我们就能得到 SSIM 指数。

$$
\text{SSIM} = \left[l(x,y)\right]^\alpha \cdot \left[c(x,y)\right]^\beta \cdot \left[s(x,y)\right]^\gamma
$$

一般而言 $\alpha=\beta=\gamma=1$。

#### Mean SSIM

在这之上，作者使用了一个基于 Gaussian Weight Function 的滑动窗口算法 (Sliding Window Algorithm) 以分别计算图片中各区域的 SSIM 并将其加权取得最终的 Mean SSIM。相关公式和具体的计算方法可以参照论文原文。

**当我们一般使用 SSIM 时，代指的都是 Mean SSIM**（如 `skimage.metrics` 中的 [`structural_similarity()`](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity) 函数）。

关于 Mean SSIM 为什么相对 SSIM 更加有效，请参照论文中的逻辑：

> For image quality assessment, it is useful to apply the SSIM index locally rather than globally. First, image statistical features are usually highly spatially nonstationary. Second, image distortions, which may or may not depend on the local image statistics, may also be space-variant. Third, at typical viewing distances, only a local area in the image can be perceived with high resolution by the human observer at one time instance (because of the foveation feature of the HVS [49], [50]). And finally, localized quality measurement can provide a spatially varying quality map of the image, which delivers more information about the quality degradation of the image and may be useful in some applications.

### 评估示例

在现实中进行评估时，多数论文仍然会同时给出 PSNR 与 SSIM 值。我们可以通过以下的示例代码对模型进行评估。

注意：如果模型放大倍数为 4x，那么需要加载 4x 的 HR 原图；此处默认模型为 2x 放大。

```python
# 加载库
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 加载低分辨率图片（用于模型推理） & 高分辨率原图（用于评估比较）
img_lr = cv2.imread(R'D:\Ye Shu\Set14\image_SRF_2\img_001_SRF_2_LR.png')
img_hr = cv2.imread(R'D:\Ye Shu\Set14\image_SRF_2\img_001_SRF_2_HR.png')

# 运行模型（model 可以为任何模型）
img_pred = model(img_lr)

# 评估模型输出
print("SSIM: " + str(
    ssim(img_hr, img_pred, 
         data_range=img_pred.max() - img_pred.min(),
         # 如果图片 channel 数大于 1（彩色图片），需要设置 multichannel=True
         multichannel=True)
))

print("PSNR: " + str(
    psnr(img_hr, img_pred)
))
```

另：同一数据集中一般会有多个图片，严谨的做法是对所有图片求得 PSNR & SSIM 后对两者分别取平均值得出结论。
