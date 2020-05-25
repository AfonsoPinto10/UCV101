import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
    

def unet(pw = None, img_s = (256,256,1)):
 
    inputs = Input(img_s)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pw):
    	model.load_weights(pw)

    return model

def vnet(pw = None, img_s =(138,128,3)): #=add size pls#
    #layer 1 down
    inputs = Input(img_s)
    conv1 = PReLU(Conv3D(16, 5, strides=1, padding = 'same', kernel_initializer = 'he_normal')(inputs))
    repeat1 = concatenate(16 * [inputs], axis=-1)
    add1 = add([conv1, repeat1]) #Não percebo esta parte do código Help
    ds1 = Conv3D(32, 2, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(add1)
    ds1=PReLU()(ds1)
    
    #layer 2 down
    conv2 = PReLU(Conv3D(32, 5, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(ds1))
    conv2 = PReLU(Conv3D(32, 5, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv2))
    add2 = add([conv2,ds1])
    ds2 = Conv3D(64, 2, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(add2)
    ds2 = PReLU()(ds2)
    
    #layer 3 down
    conv3 = PReLU(Conv3D(64, 5, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(ds2))
    conv3 = PReLU(Conv3D(64, 5, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv3))
    conv3 = PReLU(Conv3D(64, 5, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv3))
    add3 = add([conv3,ds2])
    ds3 = Conv3D(128, 2, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(add3)
    ds3 = PReLU()(ds3)
    
    #layer 4 down
    conv4 = PReLU(Conv3D(128, 5, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(ds3))
    conv4 = PReLU(Conv3D(128, 5, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv4))
    conv4 = PReLU(Conv3D(128, 5, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv4))
    add4 = add([conv4,ds3])
    ds4= Conv3D(256, 2, strides = 2, padding = 'same', kernel_initializer = 'he_normal')(add4)
    ds4= PReLU()(ds4)
    
    #Layer 5
    conv5 = PReLU(Conv3D(256, 5, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(ds4))
    conv5 = PReLU(Conv3D(256, 5, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv5))
    conv5 = PReLU(Conv3D(256, 5, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv5))
    add5 = add([conv5,ds4])
    us5 = Conv3DTranspose(256,2,strides=2, padding = 'same', kernel_initializer = 'he_normal')(add5) #UNSURE
    us5 = PReLU()(us5)
    
    #layer 6 Up
    copy6 = concatenate([us5, add4], axis=4) #Unsure
    conv6 = PReLU(Conv3D(256, 5,strides = 1, padding = 'same', kernel_initializer = 'he_normal')(copy6))
    conv6 = PReLU(Conv3D(256, 5,strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv6))
    conv6 = PReLU(Conv3D(256, 5,strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv6))
    add6 = add([conv6, copy6])
    us6 = Conv3DTranspose(128,2,strides=2, padding = 'same', kernel_initializer = 'he_normal')(add6)
    us6 = PReLU()(us6)
    
    #Layer 7 Up
    copy7 = concatenate([us6, add3], axis=4) #Unsure
    conv7 = PReLU(Conv3D(128, 5,strides = 1, padding = 'same', kernel_initializer = 'he_normal')(copy7))
    conv7 = PReLU(Conv3D(128, 5,strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv7))
    conv7 = PReLU(Conv3D(128, 5,strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv7))
    add7 = add([conv7, copy7])
    us7 = Conv3DTranspose(64,2,strides=2, padding = 'same', kernel_initializer = 'he_normal')(add7)
    us7 = PReLU()(us7)
    
    #Layer 8 up
    copy8 = concatenate([us7, add2], axis=4) #Unsure /nvm i think i got it
    conv8 = PReLU(Conv3D(64, 5,strides = 1, padding = 'same', kernel_initializer = 'he_normal')(copy8))
    conv8 = PReLU(Conv3D(64, 5,strides = 1, padding = 'same', kernel_initializer = 'he_normal')(conv8))
    add8 = add([conv8, copy8])
    us8 = Conv3DTranspose(32,2,strides=2, padding = 'same', kernel_initializer = 'he_normal')(add8)
    us8 = PReLU()(us8)
    
    #layer 9
    copy9 = concatenate([us8, add1], axis=4)
    conv9 = PReLU(Conv3D(32, 5,strides = 1, padding = 'same', kernel_initializer = 'he_normal')(copy8))
    add9 = add([conv9, copy9]) #estas adições podem estar erradas/unsure
    conv9 = PReLU(Conv3D(2, 1,strides = 1, padding = 'same', kernel_initializer = 'he_normal')(copy8)) #Unsure
    sftm = Softmax()(conv9)
    
    #Com+ile. Ainda falta colocar a função de loss indicada
    model = Model(inputs = inputs, outputs = sftm)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pw):
    	model.load_weights(pw)

    return model
    