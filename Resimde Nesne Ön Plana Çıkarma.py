import numpy as np
import cv2
from matplotlib import pyplot as plt

img=cv2.imread('on_plan.jpg')
mask=np.zeros(img.shape[:2],np.uint8)#ilk iki elemanı alır

bgdMmodel=np.zeros((1,65),np.float64)#bu ikisi algoritmada tanımlanmalı
fgdMmodel=np.zeros((1,65),np.float64)

diktortgen=(250,125,190,450)

cv2.grabCut(img,mask,diktortgen,bgdMmodel,fgdMmodel,5,cv2.GC_INIT_WITH_RECT)
mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
img=img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()