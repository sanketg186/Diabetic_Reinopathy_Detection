import torch.utils.data as data
from torchvision import transforms
import csv
import pandas as pd
import os
from PIL import Image
import numpy as np
import random

def acc(y_act,y_pred):
	if(y_act==y_pred):
		return 1
	elif y_act>0 and y_pred==1:
		return 1
	else :
		return 0



class CDATA(data.Dataset):
	def __init__(self,csv_path,img_path,transform=None):
		df=pd.read_csv(csv_path)
		#print(df.head())
		#print(df['train_image_name'])
		self.transform=transform
		self.img_path=img_path
		self.X_train=df['train_image_name']
		self.y_train=df['level']

	def __getitem__(self,index):
		#print(X_train[index])
		img=Image.open(self.img_path+self.X_train[index])
		if self.y_train[index]>0:
			label=1.0
		else:
			label=0.0

		img=img.convert('RGB')

		if self.transform is not None:
			img=self.transform(img)


		return img,label

	def __len__(self):
		return len(self.X_train.index)
