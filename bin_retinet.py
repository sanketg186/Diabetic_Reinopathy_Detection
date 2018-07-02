
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from torch.utils.data.sampler import RandomSampler, SequentialSampler
##import img.transformer as transformer
import csv
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch


class RetiNet(nn.Module):
	def __init__(self):
		super(RetiNet,self).__init__()
		self.cnn1=nn.Sequential(
			nn.Conv2d(3,32,kernel_size=(4,4),stride=1,padding=1),
			 nn.BatchNorm2d(32),
			 nn.ReLU()
			)
		self.cnn2=nn.Sequential(
			nn.Conv2d(32,32,kernel_size=(4,4),stride=1,padding=1),
			 nn.BatchNorm2d(32),
			 nn.ReLU()
			)
		self.cnn3=nn.Sequential(
			nn.Conv2d(32,32,kernel_size=(4,4),stride=2,padding=1),
			 nn.BatchNorm2d(32),
			 nn.ReLU()
			)

		self.cnn4=nn.Sequential(
			nn.Conv2d(32,32,kernel_size=(4,4),stride=1,padding=1),
			 nn.BatchNorm2d(32),
			 nn.ReLU()
			)

		self.cnn5=nn.Sequential(
			nn.Conv2d(32,32,kernel_size=(4,4),stride=1,padding=1),
			 nn.BatchNorm2d(32),
			 nn.ReLU()
			)
		self.cnn6=nn.Sequential(
			nn.Conv2d(32,32,kernel_size=(4,4),stride=1,padding=1),
			 nn.BatchNorm2d(32),
			 nn.ReLU()
			)

		self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
		self.cnn7=nn.Sequential(
			nn.Conv2d(32,32,kernel_size=(4,4),stride=1,padding=1),
			 nn.BatchNorm2d(32),
			 nn.ReLU()
			)
		self.cnn8=nn.Sequential(
			nn.Conv2d(32,32,kernel_size=(4,4),stride=1,padding=1),
			 nn.BatchNorm2d(32),
			 nn.ReLU()
			)
		self.fcn1 =nn.Sequential( 
			nn.Linear(60*60*32,512),
			nn.Sigmoid(),
			nn.Dropout(.25)
			)
		self.fcn2 =nn.Sequential( 
			nn.Linear(512,256),
			nn.Sigmoid(),
			nn.Dropout(.25)
			)
		self.fcn3 =nn.Sequential( 
			nn.Linear(256,1),
			nn.Sigmoid(),
			)
		self.sigmoid=nn.Sigmoid()

	def forward(self,x):
		#print(x.size())
		x=self.cnn1(x)
		#print(x.size())
		x=self.cnn2(x)
		#print(x.size())
		x=self.cnn3(x)
		x=self.cnn4(x)
		x=self.cnn5(x)
		x=self.cnn6(x)
		
		#print(x.size())
		#x=self.cnn4(x)
		#x=self.cnn5(x)
		#print("cnn=",x.size())
		x=self.maxpool(x)
		x=self.cnn7(x)
		x=self.cnn8(x)
		#print(x.size())
		#print(x.size())
		#print("maxpool",x.size())
		x=self.fcn1(x.view(1,-1))
		x=self.fcn2(x)
		x=self.fcn3(x)
		#x=self.fcn3(x)
		#x=self.fcn4(x)
		#x=self.sigmoid(x)
		return x
