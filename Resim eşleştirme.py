import numpy as np
import cv2
import matplotlib.pyplot as plt
resim_aranacak=cv2.imread('kucuk_resim.JPG',0)
resim_buyuk=cv2.imread('buyuk_resim.JPG',0)

orb=cv2.ORB_create()#eşleştirmeleri bunla yapıcaz
an1,hedef1=orb.detectAndCompute(resim_aranacak,None)
an2,hedef2=orb.detectAndCompute(resim_buyuk,None)
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)#ortak noktaları bulmak için
eslesmeler=bf.match(hedef1,hedef2)#hedef 1 le hedef 2 yi karşılaştır
eslesmeler=sorted(eslesmeler,key= lambda x:x.distance)#eşleştirmeleri sırala
son_resim=cv2.drawMatches(resim_aranacak,an1,resim_buyuk,an2,eslesmeler[:10],None,flags=2)#eşleşmeleri gösteren çizgi yapar
plt.imshow(son_resim)
plt.show()