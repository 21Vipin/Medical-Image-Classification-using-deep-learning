from __future__ import print_function
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
import numpy as np
import pickle
import json
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
from keras import metrics
import sys
import math

from Models import new
from Models import ZFNet

def main():
	path=sys.argv[1]
	with open(path) as f:
		config=json.load(f)
	nb_epochs=int(config['epochs'])
	batch_size=int(config['batch_size'])
	nb_classes=int(config['nb_classes'])
	pre_train=None
	#tbCallBack = TensorBoard(log_dir='Graph', histogram_freq=0, write_graph=True, write_images=True)
	if config['pre_train']=='True':
		pre_train=True
	else:
		pre_train=False
	shuffle=None
	if config['shuffle']=='True':
		shuffle=True
	else:
		shuffle=False
	weight_path=config['weights']
	model=None
	if pre_train:
		model=new.load_model(nb_classes,weight_path)
	else:
		model=new.load_model(nb_classes)

	sgd =  SGD(lr=0.00005, decay = 1e-5, momentum=0.99, nesterov=True) # 1e-6 = 10^-6
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	#model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=[metrics.mae, metrics.sparse_categorical_accuracy])
	
	###################################for first level calculation #######################
	train={}
	with open(config['data_path']+'/'+config['dataset_name']+'.train','rb') as f:
		train=pickle.load(f)

	#######################################################################################
	
	############ for second level classification#########################
	# train={}
	# with open(config['data_path']+'/'+config['dataset_name']+'.further','rb') as f:
	# 	train=pickle.load(f)

	###################################################################################

	x_train,y_train,imgname=train['data'],train['labels'],train['imgname']
	x_train = x_train.reshape((-1,227,227,1))
	#y_train=y_train.reshape((-1,1))
	#print(y_train)


	class LossHistory(Callback):
		def on_train_begin(self, logs={}):
			self.losses = []

		def on_batch_end(self, batch, logs={}):
			self.losses.append(logs.get('loss'))


	print(x_train.shape, 'train samples')
	print(y_train.shape, 'train sample labels')
	#https://gis.stackexchange.com/questions/72458/export-list-of-values-into-csv-or-txt-file
	checkpointer = ModelCheckpoint(filepath=config['weights'], verbose=1, save_best_only=True)
	fit_1=model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epochs, validation_split=config['validation_split'], shuffle=shuffle, verbose=1, callbacks=[checkpointer]) #,callbacks=tbCallback)
	print(history.losses)


	# import csv
	# csvfile = "history.csv"

	#Assuming res is a flat list
	# with open(csvfile, "w") as output:
	#     writer = csv.writer(output, lineterminator='\n')
	#     for val in fit_1:
	#         writer.writerow([val])


	#model.save_weights(config['weights'])
	#tbCallback.set_model(model)
	print(fit_1.history.keys())
	#print(fit_1.history.keys())
	# summarize history for accuracy
	plt.plot(fit_1.history['acc'])
	plt.plot(fit_1.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.figure(0)
	plt1.plot(fit_1.history['loss'])
	plt1.plot(fit_1.history['val_loss'])
	plt1.title('model loss')
	plt1.ylabel('loss')
	plt1.xlabel('epoch')
	plt1.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt1.show()

if __name__ == '__main__':
	if len(sys.argv)==1:
		print("Please include the config.json file path like this - python train.py config.json")
	else:
		main()
