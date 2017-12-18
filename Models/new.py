from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def load_model(nb_classes=1000,path_to_weights=None):
    model = Sequential()
    model.add(Convolution2D(32,5,5,border_mode="valid",subsample=(2,2),input_shape=(227,227,1))) #output=((227-5)/2 + 1 = 112
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((112-2)/2 + 1 = 56
    

    model.add(Convolution2D(32,5,5,border_mode="same")) #output = 56
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3,border_mode="same"))  #output = 56
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((56-2)/2 + 1 = 28


    
    model.add(Convolution2D(64,3,3,border_mode="same"))  #output = 28
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3,border_mode="same")) #output= 28
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((28-2)/2 + 1 = 14
    
    
    
    model.add(Convolution2D(96,3,3,border_mode="same"))  #output = 14
    model.add(BatchNormalization())
    model.add(Activation('relu'))    
    model.add(Convolution2D(96,3,3,border_mode="valid"))  #output = ((14-3)/1) +1 = 12
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((12-2)/2 + 1 = 6
    
    

    model.add(Convolution2D(192,3,3,border_mode="same"))  #output =6
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(192,3,3,border_mode="valid"))  #output = ((6-3)/1) + 1 = 4 
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((4-2)/2 + 1 = 2 
    
    model.add(Flatten())
    
    model.add(Dense(output_dim=4096,input_dim=2*2*192))
    model.add(Activation('relu'))
    #model.add(Dropout(0.4)) # for first level
    model.add(Dropout(0.4)) # for sec level
    
    model.add(Dense(output_dim=4096,input_dim=4096))
    model.add(Activation('relu'))
    #model.add(Dropout(0.4)) # for first level
    model.add(Dropout(0.4)) # for sec level
    
    model.add(Dense(output_dim=nb_classes,input_dim=4096))
    model.add(Activation('softmax'))
    
    if not path_to_weights==None:
        model.load_weights(path_to_weights)
    
    return model

