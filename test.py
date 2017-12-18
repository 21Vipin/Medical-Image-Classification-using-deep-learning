from __future__ import print_function
from keras.optimizers import SGD
import numpy as np
import pickle
import json
#from keras import metrics
#import os
import sys
from pandas_confusion import ConfusionMatrix

from Models import new # model for first level classification
from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib
#import brewer2mpl
import numpy as np
import pandas as pd
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam
import cv2

from keras.preprocessing.image import img_to_array
from vis.utils.utils import stitch_images








def main():
	path=sys.argv[1]
	with open(path) as f:
		config=json.load(f)
	batch_size=int(config['batch_size'])
	nb_classes=int(config['nb_classes'])
	weight_path=config['weights']

	
	
	

	#####################First level of Classification ################################

	##### load model
	model=None

	model=new.load_model(nb_classes,weight_path)
		


	####### specify the loss function
	sgd = SGD(lr=0.00005, decay = 1e-5, momentum=0.99, nesterov=True)
	#sgd = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	#model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=[metrics.mae,  metrics.sparse_categorical_accuracy])



	######## load data
	test={}
	with open(config['data_path']+'/'+config['dataset_name']+'.test','rb') as f:
		test=pickle.load(f)
	
	x_test,y_test,imgname=test['data'],test['labels'],test['imgname']
	x_ts = x_test.reshape((-1,227,227,1))
	
	print(x_ts.shape, 'test samples')
	print(y_test.shape, 'test sample labels')



	##### evalution and prediction and confusion matrix formation
	scores=model.evaluate(x_ts,y_test,batch_size=batch_size,verbose=0)
	print("model %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	prediction= model.predict_classes(x_ts,verbose=1)
	#print(prediction)
	np.save('prediction.npy', prediction)
	pre=np.array(prediction)
	pre=MultiLabelBinarizer().fit_transform(pre.reshape(-1, 1))
	orig=y_test
	print('')
	print('')
	print('score for first level classification:   ',scores)
	'''
	count = 0
	for i in range(0,len(pre)):
		if not np.array_equal(orig[i],pre[i]):
			print(imgname[i],"_",orig[i],"_",pre[i],"_False")
			count = count + 1 
	print (count)
	'''
	aa=[0,1]
	aa = np.array(aa)
	print('')
	print('')
	print(MultiLabelBinarizer().fit_transform(aa.reshape(-1, 1)))
	print("0-Nontumor    1-Tumor")
	a=[0,1]
	a=np.array(a)
	b=[1,0]
	b=np.array(b)
	y_true = []
	y_pred = []
	print(range(len(prediction)))

	for i in range(len(prediction)):
		if np.array_equal(orig[i],a):
			y_true.append(1)
		elif np.array_equal(orig[i],b):
			y_true.append(0)
		
	for i in range(len(prediction)):
		if np.array_equal(pre[i],a):
			y_pred.append(1)
		elif np.array_equal(pre[i],b):
			y_pred.append(0)

	cm = ConfusionMatrix(y_true, y_pred)
	print('')
	print('')
	print('*****************************Confusion Matrix for first level Classification****************************')
	print(cm)

	print('')

	print('')
	



############################ Second level Classification ###############################
	'''
	path=sys.argv[2]
	with open(path) as f:
		config2=json.load(f)
	batch_size2=int(config2['batch_size'])
	nb_classes2=int(config2['nb_classes'])
	weight_path2=config2['weights']


	##### load model2
	model2=None

	model2=new.load_model(nb_classes2,weight_path2)
		


	####### specify the loss function
	sgd2 = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
	model2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	#model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=[metrics.mae,  metrics.sparse_categorical_accuracy])



	######## load data
	test={}
	with open(config2['data_path']+'/'+config2['dataset_name']+'.further','rb') as f:
		test2=pickle.load(f)
	
	x_test2,y_test2,imgname2=test2['data'],test2['labels'],test2['imgname']
	x_ts2 = x_test2.reshape((-1,227,227,1))



	count=0
	tumorname = []
	for i in range(0,len(pre)):
		if np.array_equal(pre[i],a):
			tumorname.append(imgname[i])
			count+=1

	print(count)
	print(len(tumorname))
	print(len(imgname2))
	tumor = []
	tumorlabels = []
	count=0
	count1=0
	for i in range(len(tumorname)):
		for j in range(len(imgname2)):
			if tumorname[i] == imgname2[j]:
				tumor.append(x_ts2[j])
				tumorlabels.append(y_test2[j])
				count+=1

	print(count)
	
	tumor = np.array(tumor)
	tumorlabels = np.array(tumorlabels)
	overview(0,207, tumor)
	
	print(tumor.shape, ' predicted tumor samples')
	print(tumorlabels.shape, 'predicted tumor sample labels')
	print('')


	##### evalution and prediction and confusion matrix formation
	scores2=model2.evaluate(tumor,tumorlabels,batch_size=batch_size2,verbose=0)
	print("model2 %s: %.2f%%" % (model2.metrics_names[1], scores2[1]*100))
	prediction2= model2.predict_classes(tumor,verbose=1)
	#print(prediction)
	np.save('prediction2.npy', prediction)
	pre2=np.array(prediction2)
	pre2=MultiLabelBinarizer().fit_transform(pre2.reshape(-1, 1))
	orig2=tumorlabels
	print('')
	print('')
	print('score for second level classification',scores2)
	count2 = 0
	aa=[0,1,2,3]
	aa = np.array(aa)
	print('')
	print(aa,MultiLabelBinarizer().fit_transform(aa.reshape(-1, 1)))
	print("0-astrocytoma  1-gbm   2- mixed   3- oligodendroglioma")
	print('')
	a=[1,0,0,0]
	b=[0,1,0,0]
	c=[0,0,1,0]
	d=[0,0,0,1]

	a=np.array(a)
	b=np.array(b)
	c=np.array(c)
	d=np.array(d)
	y_true2 = []
	y_pred2 = []


	for i in range(len(prediction2)):
		if np.array_equal(orig2[i],a):
			y_true2.append(0)
		elif np.array_equal(orig2[i],b):
			y_true2.append(1)
		elif np.array_equal(orig2[i],c):
			y_true2.append(2)
		elif np.array_equal(orig2[i],d):
			y_true2.append(3)

	for i in range(len(prediction2)):
		if np.array_equal(pre2[i],a):
			y_pred2.append(0)
		elif np.array_equal(pre2[i],b):
			y_pred2.append(1)
		elif np.array_equal(pre2[i],c):
			y_pred2.append(2)
		elif np.array_equal(pre2[i],d):
			y_pred2.append(3)

	cm2 = ConfusionMatrix(y_true2, y_pred2)
	print('')
	print('*****************************Confusion Matrix for SECOND level Classification****************************')
	print(cm2)
#	cm2.print_stats()
	print('')


	counter=0
	for i in range(len(pre2)):
		if np.array_equal(pre2[i],a):
			#print(tumorname[i],'__',a,'__astrocytoma')
			counter+=1

	print('')
	print(counter,'__astrocytoma__images')
	counter=0
	for i in range(len(pre2)):
		if np.array_equal(pre2[i],b):
			#print(tumorname[i],'__',b,'__gbm')
			counter+=1
	
	print('')
	print(counter,'__gbm__images')
	counter=0
	for i in range(len(pre2)):
		if np.array_equal(pre2[i],c):
			#print(tumorname[i],'__',c,'__mixed')
			counter+=1
	
	print('')
	print(counter,'__mixed__images')
	counter=0
	for i in range(len(pre2)):
		if np.array_equal(pre2[i],d):
			#print(tumorname[i],'__',d,'__oligodendroglioma')
			counter+=1

	print('')
	print(counter,'__oligodendroglioma__images')

	# layer_name = 'predictions'
	# layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

	# Images corresponding to tiger, penguin, dumbbell, speedboat, spider
	

	# heatmaps = []
	# for img in tumor:
	#     # Predict the corresponding class for use in `visualize_saliency`.
	#     pred_class = np.argmax(model.predict(np.array([img_to_array(img)])))

	#     # Here we are asking it to show attention such that prob of `pred_class` is maximized.
	#     heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img, text=tumorlabels)
	#     heatmaps.append(heatmap)

	# cv2.imwrite('predictions.png',stitch_images(heatmaps))
	'''

if __name__ == '__main__':
	if len(sys.argv)==1:
		print("Please include the config.json file path like this - python train.py config.json")
	else:
		main()