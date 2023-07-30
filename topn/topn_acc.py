import os
import time
import sys

class Logger(object):
    def __init__(self, filename="Default.log", path="./"):
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, filename), "a", encoding='utf8',)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def txtPath(resPath:str)->list:
    path=os.path.abspath(resPath)
    l=os.walk(path)
    fileList=[]
    for home,dirs,files in l:
        for filename in files:
            fileList.append(os.path.join(home, filename))
    return fileList

def getLable(lablePath):
    lablePath=os.path.abspath(lablePath)
    lableList=[]
    with open(lablePath,'r') as f:
        for l in f.readlines():
            lableList.append(int(l))
    return lableList

def getRes(oneResPath):
    oneResList=[]
    with open(oneResPath,'r') as f:
        for l in f.readlines():
            oneResList.append(l)
    return oneResList

def acc(resPath,lablePath):
    lableList=getLable(lablePath)
    fileList=txtPath(resPath)
    # print("fileList is : " + str(fileList))
    totalNum=len(fileList)
    top1_counter=0
    topn_counter=0
    for i in range(totalNum):
        oneResPath=fileList[i]
        oneRes=getRes(oneResPath)
        oneResl=[]
        for r in oneRes:
            r=r.split()
            oneResl.append(int(r[0]))
        if oneResl[0]==lableList[i]:
            top1_counter+=1
            topn_counter+=1
        elif lableList[i] in oneResl:
            topn_counter+=1
    top1=top1_counter/totalNum
    topn=topn_counter/totalNum
    print(f"name: {resPath}, top1: {top1}, top{len(oneRes)}: {topn}")

def getLog(log_root):
    log_root=os.path.abspath(log_root)
    if not os.path.exists(log_root):
        print(f'\033[31;43mwarning: log path is not exist: create new path <{log_root}> \033[0m')
        os.mkdir(log_root)
    log_time = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    sys.stdout = Logger(log_root+'//'+log_time+'.log')


if __name__=="__main__":
    modellist=[r'E:\GitLab\metrics\classify\EfficientNet_restemp\efficientnet_pt_16_sim']   # 需要手动修改路径 上板结果存放路径
    lablePath='5000labels.txt'
    log_root='./classlog'
    getLog(log_root)
    for resPath in modellist:
        print("resPath is : " + str(resPath))
        acc(resPath,lablePath)
    
    