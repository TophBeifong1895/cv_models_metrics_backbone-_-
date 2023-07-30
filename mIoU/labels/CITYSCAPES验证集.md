# CITYSCAPES验证集

目录结构：

```bash
├───gtFine
│   └───val
│       ├───frankfurt
│       ├───lindau
│       └───munster
└───leftImg8bit
    └───val
        ├───frankfurt
        ├───lindau
        └───munster
```

**gtFine：**存放的是ground-truth，名称中的Fine表示的是精确标注的ground truth。Cityscapes还定义了一套粗标注的数据集，那个目前用不着。gtFine下面有一个val文件夹，存放的是验证集。（训练集和测试集被我删了，因为用不到）

val中的3个文件夹代表了在法国的3个城市拍摄得到的图片数据。验证集里是法兰克福、林道、蒙斯特。

在测试集的ground truth中，每一张图片都会对应6个文件，以法兰克福中的第一张图片为例：

* frankfurt_000000_000294_gtFine_color.png
* frankfurt_000000_000294_gtFine_instanceIds.png
* frankfurt_000000_000294_gtFine_instanceTrainIds.png
* frankfurt_000000_000294_gtFine_labelIds.png
* frankfurt_000000_000294_gtFine_labelTrainIds.png
* frankfurt_000000_000294_gtFine_polygons.json

Cityscape中的图片命名是用下划线"_"将名称分为四部分。

第一部分：城市名，即对应frankfurt、lindau、munster；

第二部分：图片序号，图片序号又分为两个小序列：

- 在frankfurt中，前6位为000000和000001，后6位没有规律，共计267张图片；
- 在lindau中，前6位从000000到000058，后65位固定000019，共计59张图片；
- 在munster中，前6位从000000到00173，后6位固定000019，共计174张图片。

第三部分：gtFine，表示是精标注的ground truth；

第四部分：文件属性，具体来说：

- color：大小为$1024 \times 2048$的RGBA四通道图，可以当成一个分割样例；
- instanceIds：大小为$1024 \times 2048$的8bit灰度图，测试的时候用不到；
- instanceTrainIds：大小为$1024 \times 2048$的8bit灰度图，测试的时候用不到；
- **labelIds**：大小为$1024 \times 2048$的8bit灰度图，但像素值的取值范围为0-33（总共34类）；
- **labelTrainIds**：大小为$1024 \times 2048$的8bit灰度图，像素取值范围0-18（总共19类）；
- polygons：JSON文件，内容是一个字典，总共三个key：imgHeight、imgWidth、objects。其中object是一个列表，列表元素为图片上每个类别所对应的多边形信息构成的字典。字典的key有label和polygon，其中label放的是类别名，polygen放的是多边形的定点坐标。

**leftImg8bit：**存放的就是所需的测试图片，命名规则同上。



**使用方法**

这里有一个困扰的点在于`labelIds`和`labelTrainIds`为什么对应的类别数不同，分别34和19。我们可以看下面的CitysScape的标签列表，其中`id`对应的列就是每个类别的label id，有34类。但CityScape也说了，有一些类别是不包含网络的前向推导中，也就是说在CityScape上训练的分割网络只输出19类：

> - \+ This label is not included in any evaluation and treated as void (or in the case of *license plate* as the vehicle mounted on).

那是哪19类呢？

结合下面的列表中`trainId`和`ignoreInEval`两列，`ignoreInEval`值为1的类别将不被包含在模型的前向输出中，对应的`trainId`也被设置成255。

现在回过头看`labelIds`和`labelTrainIds`对应的图片，后者中明显会出现很多白色像素，这些像素在计算精度的时候是被忽略的。

到这里可以明确，CityScape上训练的网络输出0-18。那么到底用`labelIds`还是`labelTrainIds`来测试？答案是两者都可以。

对于`labelIds`，由于网络输出的0-18并不就是数据集类别标签的前19个，所以需要做一个简单的序号映射，下面`id`和`trainId`两列可以清楚看到两者的映射关系。例如网络输出0，对应的是road，而0对应标签7。那么剩下的其他类别呢？全部换成255就行。所以，使用`labelIds`进行测试的时候，需要对`labelIds`对应的图片进行一次“预处理”，操作的内容就是完成像素值的映射，即把图中像素值从列表中的列`id`映射成列`trainId`。

对于`labelTrainIds`，由于标签已经对上了，所以直接计算相应的精度指标即可。

```python

                     name |  id | trainId |       category | categoryId | hasInstances | ignoreInEval|        color
    --------------------------------------------------------------------------------------------------
                unlabeled |   0 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
              ego vehicle |   1 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
     rectification border |   2 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
               out of roi |   3 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
                   static |   4 |     255 |           void |          0 |            0 |            1 |         (0, 0, 0)
                  dynamic |   5 |     255 |           void |          0 |            0 |            1 |      (111, 74, 0)
                   ground |   6 |     255 |           void |          0 |            0 |            1 |       (81, 0, 81)
                     road |   7 |       0 |           flat |          1 |            0 |            0 |    (128, 64, 128)
                 sidewalk |   8 |       1 |           flat |          1 |            0 |            0 |    (244, 35, 232)
                  parking |   9 |     255 |           flat |          1 |            0 |            1 |   (250, 170, 160)
               rail track |  10 |     255 |           flat |          1 |            0 |            1 |   (230, 150, 140)
                 building |  11 |       2 |   construction |          2 |            0 |            0 |      (70, 70, 70)
                     wall |  12 |       3 |   construction |          2 |            0 |            0 |   (102, 102, 156)
                    fence |  13 |       4 |   construction |          2 |            0 |            0 |   (190, 153, 153)
               guard rail |  14 |     255 |   construction |          2 |            0 |            1 |   (180, 165, 180)
                   bridge |  15 |     255 |   construction |          2 |            0 |            1 |   (150, 100, 100)
                   tunnel |  16 |     255 |   construction |          2 |            0 |            1 |    (150, 120, 90)
                     pole |  17 |       5 |         object |          3 |            0 |            0 |   (153, 153, 153)
                polegroup |  18 |     255 |         object |          3 |            0 |            1 |   (153, 153, 153)
            traffic light |  19 |       6 |         object |          3 |            0 |            0 |    (250, 170, 30)
             traffic sign |  20 |       7 |         object |          3 |            0 |            0 |     (220, 220, 0)
               vegetation |  21 |       8 |         nature |          4 |            0 |            0 |    (107, 142, 35)
                  terrain |  22 |       9 |         nature |          4 |            0 |            0 |   (152, 251, 152)
                      sky |  23 |      10 |            sky |          5 |            0 |            0 |    (70, 130, 180)
                   person |  24 |      11 |          human |          6 |            1 |            0 |     (220, 20, 60)
                    rider |  25 |      12 |          human |          6 |            1 |            0 |       (255, 0, 0)
                      car |  26 |      13 |        vehicle |          7 |            1 |            0 |       (0, 0, 142)
                    truck |  27 |      14 |        vehicle |          7 |            1 |            0 |        (0, 0, 70)
                      bus |  28 |      15 |        vehicle |          7 |            1 |            0 |      (0, 60, 100)
                  caravan |  29 |     255 |        vehicle |          7 |            1 |            1 |        (0, 0, 90)
                  trailer |  30 |     255 |        vehicle |          7 |            1 |            1 |       (0, 0, 110)
                    train |  31 |      16 |        vehicle |          7 |            1 |            0 |      (0, 80, 100)
               motorcycle |  32 |      17 |        vehicle |          7 |            1 |            0 |       (0, 0, 230)
                  bicycle |  33 |      18 |        vehicle |          7 |            1 |            0 |     (119, 11, 32)
            license plate |  -1 |      -1 |        vehicle |          7 |            0 |            1 |       (0, 0, 142)
```





**补充：**

1. 在使用PIL打开以上图片的时候需要注意对应的读图模式，详见下表。

| mode参数 | 含义                           |
| -------- | ------------------------------ |
| 1        | 位图，1bit                     |
| L        | 灰度图，8bit                   |
| I        | 像素为int32                    |
| F        | 像素为float32                  |
| P        | 8bit，映射为其他模式（没研究） |
| RGB      | 真色彩，3通道                  |
| RGBA     | A为alpha通道，表示透明度       |
| CMYK     | 印刷，4通道                    |
| YCbCr    | 亮色分离，3通道                |

2. 使用torchvision.datasets来读取

   torchvision.datasets支持了很多常用的数据集，预设了文件目录和文件名，读起来比较方便。

   ```python
   data = torchvision.datasets.Cityscapes(root, split, mode, target_type)
   # root: 数据集目录
   # split: 'val'
   # mode: 'fine'
   # target_type: 'semantic'
   
   img, smnt = data[0]
   # img: LeftImg8bit中的原图
   # smnt: gtFine中的labelIds
   # 索引值0-499，对应val中的500张图
   ```

   

