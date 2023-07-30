import motmetrics as mm
import os
import sys
import time
from loguru import logger

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

# get videos for test from a txt file (same as runtimeapp)
def get_videos_names(video_list_txt_path):
    with open(video_list_txt_path, 'r') as f:
        video_names = f.read().splitlines()
    return video_names

train_root = r"./labels"
result_root = r"E:\GitLab\metrics\compile_workspace\results_onboard\bytetrack_16_pc_0613"   # 需要手动修改 上板结果路径
video_names_txt_path = r".\MOT17-half-list.txt"
train_list = os.listdir(train_root)

# we only calculate metrics of videos which we test on icraft
video_names_test = get_videos_names(video_names_txt_path)
print(video_names_test)
gtfile_tsfile_map = {}
for video_name in video_names_test:
    # fetch gt txt file from train folder
    gtfile = os.path.join(train_root, video_name.split('_')[0], "gt_val_half.txt")
    tsfile = os.path.join(result_root, video_name + ".txt")
    gtfile_tsfile_map[gtfile] = tsfile

for key, value in zip(gtfile_tsfile_map.keys(), gtfile_tsfile_map.values()):
    print("===================")
    print(key, "\n", value, "\n")

acc = mm.MOTAccumulator()
mm.lap.default_solver = 'lap'

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():

        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

log_root='./classlog'
getLog(log_root)

for gtfile in gtfile_tsfile_map.keys():
    print("============================")
    print(gtfile)

    tsfile = gtfile_tsfile_map[gtfile]

    gt = mm.io.loadtxt(fname=gtfile, fmt='mot15-2D', min_confidence=1)
    ts = mm.io.loadtxt(fname=tsfile, fmt='mot15-2D',min_confidence=-1)

    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)

    mh = mm.metrics.create()

    summary = mh.compute_many([acc, acc.events[0:3]], 
                            metrics=mm.metrics.motchallenge_metrics, 
                            names=['full', 'part'])

    strsummary = mm.io.render_summary(summary, 
                                    formatters=mh.formatters, 
                                    namemap=mm.io.motchallenge_metric_names)
    
    print(strsummary)