import torch as th
import numpy as np
import cv2

img = cv2.imread('/home/mg/code/data/GEI_CASIA_B/gei/001/nm-01/001-nm-01-090.png',0)
print img.dtype
info = np.iinfo(img.dtype)
print info
cv2.imshow('source',img)
cv2.waitKey(0)
