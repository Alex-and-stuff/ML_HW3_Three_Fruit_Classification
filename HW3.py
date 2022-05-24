from numpy import who
from functions import *
from sklearn.decomposition import PCA
'''
1. Import train data
'''
train_data = np.zeros((SAMPLES*CLASS,DIM,DIM,DEPTH), dtype=np.uint8)
label_data = np.zeros((SAMPLES*CLASS,1), dtype=np.uint8)
fruits = ['Carambula', 'Lychee', 'Pear']

for idx, fruit in enumerate(fruits):
    train_data[idx*SAMPLES:(idx+1)*SAMPLES] = ImportPic(fruits[idx])
    label_data[idx*SAMPLES:(idx+1)*SAMPLES] = idx*np.ones((SAMPLES,1))
    # 0-489 490-979 980-1470

# Check if image is correct or not. If we wish to see the image, 
# make sure the np array is dtype=np.uint8, else, use float
# cv2.imshow('im',train_data[980])
# cv2.waitKey(0)

'''
2. PCA dimentional reduction to 2-dim
'''
pca = PCA(n_components=2)
train_reduc = pca.fit_transform(train_data.reshape(SAMPLES*CLASS,DIM*DIM*DEPTH))

'''
3. Divide data to train and validation set
'''
# Concat reduced training data with its label (0-Carambula, 1-Lychee, 2-Pear)
dataset = np.concatenate((train_reduc, label_data), axis=1)
# Shuffle the dataset
np.random.shuffle(dataset)
# Divide dataset to train_set and validation_set
train_set, validation_set = np.split(dataset, [int(SAMPLES*CLASS*0.9)], axis=0)

'''
4. Neural Network implementation using sigmoid fcn 
   as nonlinear mapping and train weights with SGD
   I.   Imput 
   II.  Hidden layer
   III. Output layer
'''




'''
References:
1. https://towardsdatascience.com/dimensionality-reduction-of-a-color-photo-splitting-into-rgb-channels-using-pca-algorithm-in-python-ba01580a1118
2. https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

'''
