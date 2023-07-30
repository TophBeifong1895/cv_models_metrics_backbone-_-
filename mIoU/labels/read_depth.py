#!/usr/bin/python

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    # assert(np.max(depth_png) > 255)
    print (depth_png[:,512])
    depth = (depth_png.astype(np.float)-1) / 256.
    depth[depth <= 0] = 0.0
    return depth

depth = depth_read('zurich_000121_000019_disparity.png')
print(depth)


depth_new = depth.reshape(2048*1024)
depth_new = list(depth_new)
print(len(depth_new))
new = ''
for i in depth_new:
    new += (str(i) + ' ')
# open('cityscape_01.txt', 'w').write(new)
# cv2.imshow('dep', depth)
# cv2.waitKey(0)

plt.imshow(depth)
plt.show()




