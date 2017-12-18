from __future__ import print_function
from Preprocess import patientconvert

import numpy as np
import pickle
import json
import os

import sys


def main():
	path=sys.argv[1]
	with open(path) as f:
		config=json.load(f)
	
	if os.path.exists(config['data_path']):
		patientconvert.run(config['dataset_name'],config['raw_path'],config['data_path'],float(config['train_test_split']))

if __name__ == '__main__':
	if len(sys.argv)==1:
		print("Please include the config.json file path like this - python train.py config.json")
	else:
		main()
