# from functions import *
from functions_batch import *
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

# Do PCA separately
pca.fit(train_data.reshape(TRAINSAMP*CLASS,DIM*DIM*DEPTH))

train_reduc = pca.transform(train_data.reshape(TRAINSAMP*CLASS,DIM*DIM*DEPTH))
test_reduc  = pca.transform(test_data.reshape(TESTSAMP*CLASS,DIM*DIM*DEPTH))
'''
3. Divide data to train and validation set
'''
# Concat reduced training data with its label (0-Carambula, 1-Lychee, 2-Pear)
train_set = np.concatenate((train_reduc, train_label), axis=1)
test_set  = np.concatenate((test_reduc, test_label), axis=1)
# Shuffle the dataset
np.random.shuffle(train_set)  # [1470,3]
np.random.shuffle(test_set)   # [495, 3]
# Separate data from label
train_label = train_set[:,-1]
train_set   = train_set[:,:-1]
test_label  = test_set[:,-1]
test_set    = test_set[:,:-1]
# Normalize the dataset
train_max = np.amax(train_set)
train_min = np.amin(train_set)
test_max  = np.amax(test_set)
test_min  = np.amin(test_set)
train_set = (train_set-train_min)/(train_max-train_min)
test_set  = (test_set-test_min)/(test_max-test_min)

train_label_o = train_label.reshape((len(train_label),1))
test_label_o  = test_label.reshape((len(test_label),1))
train_label = toOneHot(train_label, 3)
test_label  = toOneHot(test_label, 3)

all_set = np.concatenate((train_set, test_set), axis=0)
all_label = np.concatenate((train_label_o, test_label_o), axis=0)

# print(test_label_o, test_label_o.shape, test_label.shape)


'''
4. Build neural network
'''
batch_size = 5
net = network()
net.addLayer(FClayer(2,8,batch_size))
net.addLayer(ActivationLayer(sigmoid, sigmoid_prime))
net.addLayer(FClayer(8,3,batch_size))
# net.addLayer(ActivationLayer(sigmoid, sigmoid_prime))
# net.addLayer(FClayer(4,3,batch_size))
net.addLayer(SoftmaxLayer(3,3))
# net.addLayer(ActivationLayer(sigmoid, sigmoid_prime))
# net.addLayer(FClayer(3,3,batch_size))
# net.addLayer(ActivationLayer(sigmoid, sigmoid_prime))

net.lossFcn(cross_entropy, cross_entropy_prime)
net.fit(train_set, train_label, 300, batch_size, 0.04)
prediction = net.predict(test_set)
net.calculateAccuracy(prediction, test_label_o)


# net.lossFcn(mse, mse_prime)
# net.fit(train_set, train_label_o, 10, batch_size, 0.3)
# net.predictMSE(test_set, test_label)

net.plot_training_curve()
net.plotTrainData(train_set, train_label)
net.plotTrainData(test_set, test_label)
# net.drawDecisionRegion(all_set, test_set, test_label_o)
net.drawDecisionRegion(all_set, test_set, all_label)
plt.show()
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
4. https://www.youtube.com/watch?v=4qJaSmvhxi8&ab_channel=DeepLearningAI

'''
