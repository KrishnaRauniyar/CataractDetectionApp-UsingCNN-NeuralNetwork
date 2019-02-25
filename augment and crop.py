import numpy as np
import  matplotlib.pyplot as plt
from multiprocessing import Pool
import os

import keras
from scipy import misc,ndimage
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import pylab



#taking all images as a list
path1 = "CatarctEye"   #path to get original images
listing = os.listdir(path1)   #listing the directory 

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
        if i!=len(ims):
            pylab.savefig('aug/'+str(list[i]))
        pylab.close()
        
gen=ImageDataGenerator( 
          rotation_range=8,
          width_shift_range=0.3,
          height_shift_range=0.25,
          shear_range=0.2,
          zoom_range=0.1,
          channel_shift_range=9.,
          horizontal_flip=True)

list=[1,2,3,4,5,6,7,8,9,10]

for itr in listing:
    list[:] = [x + 10 for x in list]
    imag=np.expand_dims(ndimage.imread("CatarctEye/"+itr),0)
    aug_itr=gen.flow(imag)
    aug_img=[next(aug_itr)[0].astype(np.uint8) for i in range(10)]
    plots(aug_img,figsize=(10.270833,5.781),rows=2)
    
    
    
    
import numpy as np
import os
import cv2
import pylab
path="aug" ##originalimage that is need to be croped
listing=os.listdir(path)
i=1
for crop in listing:
    image = cv2.imread("aug/"+crop) ##taking images in the array form
    y=30
    x=77
    h=220
    w=290
    crop = image[y:y+h, x:x+w]
    cv2.imshow('Image', crop)       ##not needed but looks good while compiling
    cv2.imwrite('aug1/'+str(i)+'.jpg',crop) ##put your folder name where croped image are supposed to be by replacing 'test'
    cv2.destroyAllWindows()
    i+=1    

