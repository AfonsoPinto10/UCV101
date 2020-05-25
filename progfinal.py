# -*- coding: utf-8 -*-

# libraries to include
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import regionprops,label
import pickle

# Functions that are used on the program

def menu():
    print('(1) Input an image')
    print('(0) Quit')
    x=input('Answer: ')
    return x

def menubf():
    print('Do you want to improve your cut? What you want to remove?')
    print('(B) Background')
    print('(F) Foreground')
    print('(C) Continue')
    x=input('Answer: ')
    return x

def tracer():     # Function to select points in the image. Stops collecting when user clicks the scroll wheel
    pointu=[]
    dum=[]
    dum =plt.ginput(1)
    while dum != []:
        dum=plt.ginput(1)
        pointu.append(dum)
    return(pointu)

def pointmask(mask,tracerpoints,inclusion):   # Functions to add selected points to the mask either as background or foreground
    for point in tracerpoints:
        if point != []:
            (x,y) =point[0]
            x=int(x)
            y=int(y)
            if inclusion == True:
                mask[y-2:y+2,x-2:x+2]=np.ones((4,4))
            else:
                mask[y-2:y+2,x-2:x+2]=np.zeros((4,4))
    return(mask)

def removal(dumb,masc,bgd,fgd,imag):  # Function to update the mask according to user input
    #masc - mask
    #bgd - bgdModel
    #fgd - fgdModel
    #imag - img
    #return a new image               
    mask2 = np.where((masc==2)|(masc==0),0,1).astype('uint8')
    imag2 = imag*mask2[:,:,np.newaxis]
    img2=cv2.cvtColor(imag2, cv2.COLOR_BGR2RGB)
    img1=cv2.cvtColor(imag, cv2.COLOR_BGR2Luv)
    imgx=img1*0.10+img2*0.90
    m=np.max(imgx)
    imgx=imgx / m
    fig = plt.figure()
    plt.imshow(imgx)
    
    if dumb=='B' or dumb=='b':
        print('Select points in the UNWANTED region and click the middle mouse button to end selection')
        newselection=tracer()
        inclusion=False
        masc=pointmask(masc,newselection,inclusion)
        print('This will take a moment')
    elif dumb=='F' or dumb=='f' :
        print('Select points in the WANTED region and click the middle mouse button to end selection')
        newselection=tracer()
        inclusion=True
        masc=pointmask(masc,newselection,inclusion)
        print('This will take a moment')     
        
    masc, bgd, fgd = cv2.grabCut(imag,masc,None,bgd,fgd,100,cv2.GC_INIT_WITH_MASK)
    plt.close(fig)
    return(masc,bgd,fgd)

def imshower(img,img_cut):    #Function to display an image of the segmeted image in contrast to the real one
    imgx=cv2.cvtColor(img_cut, cv2.COLOR_BGR2RGB)
    imgx2=cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    
    imgx=imgx2*0.10+imgx*0.90
    m=np.max(imgx)
    imgx=imgx / m
    plt.imshow(imgx)
    plt.show()
    print('Click the image')
    no=plt.ginput(1)
    return()

def menup():
    print('Do you want to save the image? Or do you want to anlyze the properties of the region(s) created?')
    print('(S) Save Image')
    print('(P) Properties')
    print('(Q) Quit')
    x=input('Answer: ')
    return x

# Main program
image=0
plt.close()
print('Welcome to our Graph Cut application')
a=menu()
 
while True:
    if a=='1':
        x=input('File name: ')
        print('Please select 2 points that will form a rectangle')
        
        img = cv2.imread(str(x))
        
        fig=plt.figure()
        plt.imshow(img)
        ((tx,ty),(bx,by)) =plt.ginput(2)
        plt.close(fig)
        
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
        img_cut = img*mask2[:,:,np.newaxis] # first attempt to segment image using user defined rectangle
        
        plt.figure()
        imshower(img,img_cut)
        
        b=menubf()
        
        while True: # Further refinement of the segmentation by user introducing foreground and background
            plt.close()
            if b=='B' or b=='b':                
                mask,bgdModel,fgdModel=removal(b,mask,bgdModel,fgdModel,img)
                plt.close()
            elif b=='F' or b=='f':
                mask,bgdModel,fgdModel=removal(b,mask,bgdModel,fgdModel,img)
                plt.close()
            elif b=='C' or b=='c':
                break
            else:
                print('Unknown Input')
            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            img_cut = img*mask2[:,:,np.newaxis]
            imshower(img,img_cut)
            b=menubf()
            
        c=menup()
        while True:
            if c=='P' or c=='p':
                Lmask=label(mask2)
                Props=regionprops(Lmask)
                
                It="Properties of regions detected on image:"+x
                fh = open("Properties.txt", "a") 
                
                fh.write(It+"\n")
                print(It)
                
                re=0
                maxarea=0
                maxi=1
                for region in Props: #Calculates the area and perimeter of each region in the mask and saves it to Properties.txt
                    re+=1
                    Ar=region.area
                    Pe=region.perimeter
                    Rt="Region: "+str(re)
                    At="Area of the region: "+str(Ar)
                    Pt="Perimeter of the region: "+str(Pe)
                    
                    print(Rt)
                    print(At)
                    print(Pt)
                    
                    fh.write(Rt+"\n")
                    fh.write(At+"\n")
                    fh.write(Pt+"\n")
                    
                    if maxarea < Ar:  #Calculates the region with the biggest area
                        maxarea=Ar
                        maxlabel=maxi
                    maxi+=1
                    
                maskbig = np.where((Lmask==maxlabel),255,0).astype('uint8')
                
                bm=input("Save an image of the bigger region? (Y|N)") # allows to save the biggest region in the mask
                if bm == "Y" or bm=="y":
                    ss=input("File name for region and segmented image: ")
                    cv2.imwrite(ss+".png",img*maskbig[:,:,np.newaxis])
                    cv2.imwrite(ss+"mask.png",maskbig)
                fh.close()
            
            elif c=='S' or c=='s':  # Saving all regions in the mask
                ss=input("File name for mask and segmented image: ")
                mask2=mask2*255
                cv2.imwrite(ss+".png",img_cut)
                cv2.imwrite(ss+"mask.png",mask2)
                
            elif c=='Q' or c=='q':
                break
            
            else:
                print('Unknown Input')
                
            c=menup()
            
        a=menu()
        
    elif a=='0':
        break
    
    else:
        print('Unknown Input')
        a=menu()
        