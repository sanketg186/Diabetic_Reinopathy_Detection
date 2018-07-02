import torch.nn as nn
from torch import optim
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
from multi_retinet import RetiNet
from data_multi import CDATA,acc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

dtype=torch.cuda.FloatTensor


#csv_path='final_label.csv'
#data=pd.read_csv(csv_path,sep='\t')
#train,test=train_test_split(data,test_size=0.3,shuffle=True,random_state=1235)
#train.to_csv('train.csv',index=False)
#test.to_csv('test.csv',index=False)

composed_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((.5,.5,.5),(.5,.5,.5))])

train_ds = CDATA('train.csv','data/',transform = composed_transform)
train_loader = data.DataLoader(train_ds, batch_size=1,sampler=RandomSampler(train_ds))

val_ds = CDATA('test.csv','data/',transform = composed_transform)
val_loader = data.DataLoader(val_ds, batch_size=1,sampler=RandomSampler(val_ds))

retinet=RetiNet().type(dtype)
optimizer=optim.Adam(retinet.parameters(),lr=.0002)
loss=nn.CrossEntropyLoss()
print(len(train_loader))
num_epochs=10
c=0
fopen=open('loss.txt','w')
for epoch in range(num_epochs):
	for x,y in train_loader:
		#if(c==300):
		#	break
		#y_ar=torch.zeros(5)
		#y_ar[y]=1
		x_var=Variable(x.type(dtype))
		y_var=Variable(y.type(dtype))
		#print(y_var)
		score=retinet(x_var)
		#print(score.data[0])
		#print(score)
		#print(y_var)
		loss_val=loss(score,y_var.type(torch.cuda.LongTensor))
		optimizer.zero_grad()
		loss_val.backward()
		optimizer.step()
		if(c%100==0):
			print(c,score.data[0],loss_val.data[0])
			s=str(epoch)+": "+str(loss_val.data[0])+'\n'
			fopen.write(s)
		c=c+1

#test

torch.save(retinet.state_dict(), 'retinet.pt')
torch.save(retinet, 'retinet2.npy')
ac=0.0
c=0

for x,y in val_loader:
	#if c==300:
	#	break
	c=c+1
	x_var=Variable(x.type(dtype),volatile=True)
	y_var=Variable(y.type(dtype),volatile=True)
	#print(y_var)
	score=retinet(x_var)
	#print(score.size())
	#score=(score>.5).float()
	#print("score",score)
	val, label = torch.max(score,1)
	#print(y_var,label)
	if int(y_var) == int(label):
		ac=ac+1
	# if float(score)>=0.5:
	# 	ac=ac+acc(float(y_var),1.0)
	# else:
	# 	ac=ac+acc(float(y_var),0.0)

	if c%100==0:
		print(c)
	#loss_val=loss(score,y_var)
	#optimizer.zero_grad()
	#loss_val.backward()
	#optimizer.step()
	#print(loss_val.data[0])

print(ac)
print(ac/len(val_loader))
