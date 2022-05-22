import cv2 
import numpy as np
from regex import P
'''
Import train data
'''
def ImportPic(name, type = 'train'):
    data = []
    path = 'Data\Data_' + type + '\ ' + name + '_' + type
    for idx in range(2):
        im = cv2.imread(path+str(idx)+'.png')
        data.append(im)
    return data

fruits = ['Carambula', 'Lychee', 'Pear']

train_data = []
for fruit in fruits:
    print(fruit)
    train_data.append(ImportPic(fruit,'train'))


    
 
# print(rea[0].shape)
cv2.imshow('im',train_data[2])
cv2.waitKey(0)
# print(train_data[0], train_data.shape)

# cv2.imshow('im',train_data[0])



'''
References:
1. https://towardsdatascience.com/dimensionality-reduction-of-a-color-photo-splitting-into-rgb-channels-using-pca-algorithm-in-python-ba01580a1118

'''
