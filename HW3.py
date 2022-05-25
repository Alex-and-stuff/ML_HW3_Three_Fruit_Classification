from numpy import who
from functions import *
from sklearn.decomposition import PCA
'''
1. Import train data, test data
'''

fruits = ['Carambula', 'Lychee', 'Pear']

train_data  = np.zeros((TRAINSAMP*CLASS,DIM,DIM,DEPTH), dtype=np.uint8)
train_label = np.zeros((TRAINSAMP*CLASS,1), dtype=np.uint8)
for idx, fruit in enumerate(fruits):
    train_data[idx*TRAINSAMP:(idx+1)*TRAINSAMP] = ImportPic(fruits[idx])
    train_label[idx*TRAINSAMP:(idx+1)*TRAINSAMP] = idx*np.ones((TRAINSAMP,1))
    # 0-489 490-979 980-1470

test_data  = np.zeros((TESTSAMP*CLASS,DIM,DIM,DEPTH), dtype=np.uint8)
test_label = np.zeros((TESTSAMP*CLASS,1), dtype=np.uint8)
for idx, fruit in enumerate(fruits):
    test_data[idx*TESTSAMP:(idx+1)*TESTSAMP] = ImportPic(fruits[idx], data_type='test')
    test_label[idx*TESTSAMP:(idx+1)*TESTSAMP] = idx*np.ones((TESTSAMP,1))

# Check if image is correct or not. If we wish to see the image, 
# make sure the np array is dtype=np.uint8, else, use float
# cv2.imshow('im',train_data[980])
# cv2.waitKey(0)

'''
2. PCA dimentional reduction to 2-dim
'''
pca = PCA(n_components=2)
train_reduc = pca.fit_transform(train_data.reshape(TRAINSAMP*CLASS,DIM*DIM*DEPTH))
test_reduc  = pca.fit_transform(test_data.reshape(TESTSAMP*CLASS,DIM*DIM*DEPTH))
'''
3. Divide data to train and validation set
'''
# Concat reduced training data with its label (0-Carambula, 1-Lychee, 2-Pear)
train_set = np.concatenate((train_reduc, train_label), axis=1)
test_set  = np.concatenate((test_reduc, test_label), axis=1)
# Shuffle the dataset
np.random.shuffle(train_set)  # [1470,3]
np.random.shuffle(test_set)   # [495, 3]

'''
4. Neural Network implementation using sigmoid fcn 
   as nonlinear mapping and train weights with SGD
   I.   Input(1470*2) 
   II.  Hidden layer (fc layer -> activation layer -> ...)
   III. Output layer
'''




'''
References:
1. https://towardsdatascience.com/dimensionality-reduction-of-a-color-photo-splitting-into-rgb-channels-using-pca-algorithm-in-python-ba01580a1118
2. https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
3. https://ljvmiranda921.github.io/notebook/2017/02/17/artificial-neural-networks/

'''
