import os

old_txt = r"\\192.168.121.10\01_Personal\fangshuli\mIoU_test\CITYSCAPES\valImages.txt"
new_txt = r"\\192.168.121.10\01_Personal\fangshuli\mIoU_test\CITYSCAPES\val_Images.txt"

with open (old_txt,'r') as f:
    content = f.readlines()

print(content)

with open(new_txt,'w') as f:
   for line in content:
    line1 = (line.split(r'/'))[-1]
    f.writelines(line1)