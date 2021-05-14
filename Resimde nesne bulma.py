import cv2
import numpy as np



img_rgb=cv2.imread('ana_resim.jpg')
img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)

nesne=cv2.imread('template.jpg',0)

w,h=nesne.shape[::-1]# nesnenin boyutunu koyması tanımlaması için

res=cv2.matchTemplate(img_gray,nesne,cv2.TM_CCOEFF_NORMED)# ana resimde nesneye benzer şeyleri eşletirmeye çalıştırıyoruz
threshold=0.8 # yüzde 80 benzer olanları bulmaya çalış

loc=np.where(res>threshold) # bulduklarını yüzde 80 den fazla bezneyenleri tut

for n in zip(*loc[::-1]): ## bulduğumuz hepsini kare içine almak için bunu yapıyoruz
    cv2.rectangle(img_rgb,n,(n[0]+w,n[1]+h),(0,255,255),2)

cv2.imshow('bulunan nesneler',img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()


