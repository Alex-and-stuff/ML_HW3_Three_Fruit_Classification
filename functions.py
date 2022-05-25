import cv2
import numpy as np

TESTSAMP = 165
TRAINSAMP = 490
CLASS     = 3 
DIM       = 32
DEPTH     = 3

def ImportPic(name, data_type = 'train'):
    if data_type == 'train':
        sample = TRAINSAMP
    else:
        sample = TESTSAMP

    data = np.zeros((sample,DIM,DIM,DEPTH), dtype=np.uint8)
    path = f"Data/Data_{data_type}/{name}/{name}_{data_type}_"
    for idx in range(sample):
        im = cv2.imread(path+str(idx)+'.png')
        data[idx] = im    
    return data


'''
Fuctions for neural network
'''
# Activation fcn
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

# Fully connected layer 
class FClayer:
    def __init__(self, input_dim, output_dim, batch_size):
        self.input  = None
        self.output = None
        self.w = np.random.rand(input_dim, output_dim) - 0.5
        self.b = np.random.rand(batch_size, output_dim) - 0.5
    def forwardPropagation(self, input_data):
        self.input  = input_data
        self.output = np.dot(self.input, self.w) + self.b
        return self.output
    def backPropagation(self, error_dy, learning_rate):
        error_dx = np.dot(self.input.T,error_dy)
        error_dw = np.dot(error_dy,self.w.T)
        error_db = error_dy
        self.w  -= learning_rate*error_dw
        self.b  -= learning_rate*error_db
        return error_dx

class ActivationLayer:
    def __init__(self, act_fcn, act_fcn_prime):
        self.input  = None
        self.output = None
        self.activation = act_fcn
        self.activation_prime = act_fcn_prime
    def forwardPropagation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    def backPropagation(self, error_dy):
        return self.activation_prime(self.input)*error_dy
