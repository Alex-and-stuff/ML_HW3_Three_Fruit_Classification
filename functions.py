import cv2
import matplotlib.pyplot as plt
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

def toOneHot(label_array, categories):
    row, = label_array.shape
    encoded  = np.zeros((int(row), int(categories)))
    for r in range(row):
        for idx in range(categories):
            if label_array[r] == idx:
                encoded[r,idx] = 1
    return encoded


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

def softmax(X):
    return np.exp(X[0])/sum(np.exp(X[0]))

def cross_entropy(true_Y, predict_Y, limit=1e-12):
    # Cross entrpy: N is # of samples, K is # of classes, t_nk checks if 
    # sample is in the class or not, it is written as:
    # E = -sum(N)sum(K)t_nk*log(predict_y)
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    """
    predict_Y = softmax(predict_Y)
    ce = -np.sum(true_Y*np.log(predict_Y))
    # print(predict_Y, np.log(predict_Y))
    # return ce
    # print(true_Y, predict_Y,np.log(predict_Y))
    # ce = -np.sum(true_Y*np.log(predict_Y))
    # print('ce:', ce)
    return ce/float(predict_Y.shape[0])


def cross_entropy_prime(true_Y, predict_Y):
    # m = true_Y.shape[0]
    # grad = softmax(predict_Y)
    # grad[range(m),true_Y.max()] -= 1
    # grad = grad/m
    # return grad

    # print(-true_Y/predict_Y)
    # return -true_Y/predict_Y
    # https://medium.com/hoskiss-stand/backpropagation-with-softmax-cross-entropy-d60983b7b245
    return predict_Y-true_Y
    
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
        self.output = np.dot(self.input, self.w) #+ self.b
        return self.output
    def backPropagation(self, error_dy, learning_rate):
        error_dw = np.dot(self.input.T,error_dy)
        error_dx = np.dot(error_dy,self.w.T)
        error_db = error_dy
        # print('fc  bp: ',learning_rate, self.w.shape, error_dw.shape, self.b.shape, error_db.shape)
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
        # print('act bp: ', self.input.shape, error_dy.shape)
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
        self.training_curve_data = []
        for i in range(epochs):
            output_error = 0
            for batch in range(int(samples/batch_size)):
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

            # print('output error: ',output_error)

            print('epoch %d loss : %f'   % (i, output_error/(samples/batch_size)))
            self.training_curve_data.append([i, output_error/(samples/batch_size)])

        print('Training parameters:\n epochs = %d learning rate = %f' % (epochs, learning_rate))

        
    def predict(self, test_data, test_label):
        samples, _ = test_data.shape
        prediction = np.zeros((samples, 3))
        for sample in range(samples):
            input = test_data[sample]
            for layer in self.layers:
                input = layer.forwardPropagation(input)
            # print(input, input.shape)
            # print(input.argmax(axis=1),*input.argmax(axis=1))
            pos = np.argmax(input, axis=1)
            print(input)
            prediction[sample, pos[0]] = 1
        
        print(prediction)
        print(prediction[30:40])

        correct = 0
        for sample in range(samples):
            if np.array_equal(prediction[sample],test_label[sample]):
                correct += 1
        print("accuracy: ", correct/samples)
    def predictMSE(self, test_data, test_label):
        samples, _ = test_data.shape
        prediction = np.zeros((samples, 1))
        for sample in range(samples):
            input = test_data[sample]
            for layer in self.layers:
                input = layer.forwardPropagation(input)
            # print(input, input.shape)
            # print(input.argmax(axis=1),*input.argmax(axis=1))
            # pos = np.argmax(input, axis=1)
            print(input)
            prediction[sample] = np.round(input)
        print(prediction[0:-20])
        # print(prediction[30:40])

        correct = 0
        for sample in range(samples):
            if np.array_equal(prediction[sample],test_label[sample]):
                correct += 1
        print("accuracy: ", correct/samples)

    def plot_training_curve(self):
        epoch, loss = [],[]
        for i in range(len(self.training_curve_data)):
            epoch.append(self.training_curve_data[i][0])
            loss.append(self.training_curve_data[i][1])

        plt.plot(epoch,loss)
        plt.xlabel('Epoch')
        plt.title('training curve')

    def plotTrainData(self, train_set, train_label):
        plt.figure()
        f1x, f2x, f3x = [],[],[]
        f1y, f2y, f3y = [],[],[]
        for sample in range(train_set.shape[0]):

            
            if train_label[sample][0] == 1:
                f1x.append(train_set[sample][0])
                f1y.append(train_set[sample][1])
            elif train_label[sample][1] == 1:
                f2x.append(train_set[sample][0])
                f2y.append(train_set[sample][1])
            else:
                f3x.append(train_set[sample][0])
                f3y.append(train_set[sample][1])

        plt.scatter(f1x, f1y, c='red')
        plt.scatter(f2x, f2y, c='green')
        plt.scatter(f3x, f3y, c='blue')


