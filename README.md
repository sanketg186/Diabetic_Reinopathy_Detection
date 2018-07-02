# Diabetic_Reinopathy_Detection
Diabetic retinopathy detection using Pytorch

## About the dataset
The Dataset contains images of images of left and right eye.
More info can be found here:https://www.kaggle.com/c/diabetic-retinopathy-detection/data

## Data Preprocessing
As the images had noise so, i removed them by cropping the images, also there was class imbalance problem, so i removed it by data augmentation.

## About Implementation
Here I have implemented diabetic retinopathy detection on Kaggle Dataset.There are two implementation in this repository:

### 1. Binary classification:
In bin_retinet.py , the model predicts whether a person has diabetic retinopathy or not.

### 2.Multiclass classification:
In multi_retinet.py the model predicts whether a person has :
0 - No DR
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferative DR
