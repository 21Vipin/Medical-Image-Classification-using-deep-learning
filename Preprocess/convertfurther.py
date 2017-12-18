import os
import sys
import random
import pickle
import dicom
import math
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import rotate
from scipy import ndimage
from sklearn.preprocessing import MultiLabelBinarizer

def run(name,source,destination,split):
    x_train=[]
    x_test=[]
    train_imgname=[]
    y_train=[]
    y_test=[]
    test_imgname=[]

    _class_names=os.listdir(source)
    _class_names.sort()
    print (_class_names)
    _n2l={_class_names[i]:i for i in range(len(_class_names))}
    print (_n2l)
    

    if os.path.exists(destination):
        for i in range(len(_class_names)):
            images=os.listdir(source+"/"+_class_names[i])
            count=0
            for y in images:
                src=source+"/"+_class_names[i]+"/"+y
                img=dicom.read_file(src)
                img=img.pixel_array
                img=imresize(img,(227,227))
                x_train.append(img)
                y_train.append(_n2l[_class_names[i]])
                train_imgname.append(y)
                count+=1
            print(count)
            print(_class_names[i]+" included in training.")
    
    #print(y_ts)
    train=list(zip(x_train,y_train,train_imgname))

    #print(train)

    random.shuffle(train)

    x_train,y_train,train_imgname=zip(*train)

    x_train=np.array(x_train)
    y_tr=np.array(y_train)
    y_tr=MultiLabelBinarizer().fit_transform(y_tr.reshape(-1, 1))
    train_imgname=np.array(train_imgname)

    d_train={}
    d_train['data']=x_train
    d_train['labels']=y_tr
    d_train['imgname']=train_imgname
    #print(d_train['labels'])

    with open(destination+'/'+name+'.further','wb') as f:
        pickle.dump(d_train,f)
    