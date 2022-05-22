import cv2 
import numpy as np
path_to_file = 'Data\Data_train\Carambula'
im = cv2.imread(path_to_file + '\Carambula_train_' + str(0) + '.png')
gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

print(im, im.shape)
print(type(im))

a = np.zeros((2,32,32,3))
print(a.shape)
a[0] = im
print(a[0])
# print(gray_im, gray_im.shape)

# reim = im.reshape((1024*3,1))
# print('new', reim, reim.shape)
# cv2.imshow('Original',reim.reshape((32,32,3)))
# # cv2.imshow('Original',im)
# # cv2.imshow('Result',gray_im)
# cv2.waitKey(0)

'''
References:
1. https://towardsdatascience.com/dimensionality-reduction-of-a-color-photo-splitting-into-rgb-channels-using-pca-algorithm-in-python-ba01580a1118

'''
