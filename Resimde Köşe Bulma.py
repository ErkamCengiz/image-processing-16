import numpy as np
import cv2

resim = cv2.imread('kose_bulma.jpg')
griton = cv2.cvtColor(resim,cv2.COLOR_BGR2GRAY) # köşe tespiti için gritona çeviriyoruz
griton = np.float32(griton)#yine köşe tespititi için float32 tipine dönüştürüyoruz

köseler = cv2.goodFeaturesToTrack(griton,300,0.01,10)#bu işlemle nesne tespiti trafikteki araba tespiti her işlem yapılabilir burdaki sayılar uyarlanarak
köseler = np.int0(köseler) # en baştaki haline dönüştürüyoruz


for köse in köseler:
    x,y=köse.ravel()#bu sayede xy bileşenlerini alıyoruz direk tüm tespit edilmiş köşelerin 
    cv2.circle(resim,(x,y),3,255,-1)#tespit edilen köşelerde daireler çizdiriyoruz

cv2.imshow('köseler',resim)
cv2.waitKey(0)
cv2.destroyAllWindows()

