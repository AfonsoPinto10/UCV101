# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:22:15 2019

@author: UX401
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import regionprops


def tracer():
        print('Select points in the unwanted region and click the middle mouse button to end selection')
        pointu=[]
        dum=[]
        dum =plt.ginput(1)
        while dum != []:
            dum=plt.ginput(1)
            pointu.append(dum)
        return(pointu)

def pointmask(mask,tracerpoints):
    for point in tracerpoints:
        if point != []:
            [(x,y)] =point
            x=int(x)
            y=int(y)
            mask[y-2:y+2,x-2:x+2]=np.zeros((4,4))
    return(mask)

x=input('File name: ')

#Caso queiras fazer com imagens dicom:
#Não te esqueças de instalar a pydicom

#O comando para instalares é:
# conda install pydicom

#ds=dicom.dcmread(str(x))
#img=ds.pixel_array*128


img = cv2.imread(str(x))
ysize=img.shape[0]
plt.imshow(img)
plt.show()
((tx,ty),(bx,by)) =plt.ginput(2)
mask = np.zeros(img.shape[:2],np.uint8)  

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

tx=int(tx)
ty=int(ty)
bx=int(bx)
by=int(by)
rect = (tx,ty,bx,by)


cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)


mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

img_cut = img*mask2[:,:,np.newaxis]

plt.subplot(211),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(img_cut)
plt.title('Grab cut'), plt.xticks([]), plt.yticks([])
plt.show()
plt.close()

imgx=img+img_cut*1.20
m=np.max(imgx)
imgx=imgx / m
plt.imshow(imgx)
plt.show()
[(x,y)] =plt.ginput(1)
x=int(x)
y=int(y)
mask[ysize-y,x]=0

mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
plt.close()
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img2 = img*mask2[:,:,np.newaxis]
plt.show()
plt.imshow(img2),plt.colorbar(),plt.show()

newselection=tracer()
mask=pointmask(mask,newselection)

mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,100,cv2.GC_INIT_WITH_MASK)
plt.close()
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img2 = img*mask2[:,:,np.newaxis]
plt.imshow(img2),plt.show()

Props=regionprops(mask2)
Props[0].area
Props[0].perimeter
