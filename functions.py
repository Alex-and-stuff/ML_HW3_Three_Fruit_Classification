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

# Loss fcn
def mse(label, predict):
    return np.mean(np.power(label-predict, 2))

def mse_prime(label, predict):
    return 2*(predict-label)/label.size
    
# Fully connected layer 
class FClayer:
    def __init__(self, input_dim, output_dim, batch_size):
        self.input  = None
        self.output = None
        self.w = np.random.rand(input_dim, output_dim) - 0.5
        self.b = np.random.rand(batch_size, output_dim) - 0.5
        self.name = "FC layer"
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
        self.name = 'Activation layer'
    def forwardPropagation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output
    def backPropagation(self, error_dy, learning_rate = 0):
        return self.activation_prime(self.input)*error_dy
'''
FClayer (input [batch_size, 2(2 features)])
ActivationLayer 
FClayer (hidden layer)
ActivationLayer

for all epochs:
    for all batches:
        1.forward propagation through the layers
        2.compute the error loss
        3.back propagate the error loss and adjust weight and biases w/ SGD
'''
class network:
    def __init__(self):
        self.loss = None
        self.loss_prime = None
        self.layers = []

    def lossFcn(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
    
    def addLayer(self, layer):
        self.layers.append(layer)

    def showStructure(self):
        print("==== Network Structure ====")
        for i in range(len(self.layer)):
            print(f"Layer {str(i)}: {self.layer[i].name}")
        print("===========================")

    def fit(self, train_data, train_label, epochs, batch_size, learning_rate):
        samples,_ = train_data.shape
        for _ in range(epochs):
            output_error = 0
            for batch in range(samples/batch_size):
                input = train_data[batch*batch_size:(batch+1)*batch_size]
                # Forward propagation
                for layer in self.layers:
                    input = layer.forwardPropagation(input)

                # Compute loss fcn
                output_error += self.loss(train_label[batch*batch_size:(batch+1)*batch_size], input)

                # Back Propagation
                error = self.loss_prime(train_label[batch*batch_size:(batch+1)*batch_size], input)
                for layer in reversed(self.layers):
                    error = layer.backPropagation(error, learning_rate)


        
    # def predict(self):

