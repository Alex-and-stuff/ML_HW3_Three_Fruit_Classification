import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# def cross_entropy(true_Y, predict_Y, limit=1e-12):
#     # Cross entrpy: N is # of samples, K is # of classes, t_nk checks if 
#     # sample is in the class or not, it is written as:
#     # E = -sum(N)sum(K)t_nk*log(predict_y)
#     """
#     Computes cross entropy between targets (encoded as one-hot vectors)
#     and predictions.
#     """
#     predict_Y = softmax(predict_Y)
#     ce = -np.sum(true_Y*np.log(predict_Y))
#     output = ce/float(predict_Y.shape[0])

#     return output

# def cross_entropy_prime(true_Y, predict_Y):
#     return (predict_Y-true_Y)
def cross_entropy(true_Y, predict_Y, limit=1e-12):
    ce = -np.sum(true_Y*np.log(predict_Y))
    return ce

def cross_entropy_prime(true_Y, predict_Y):
    # print(-true_Y/predict_Y)
    return -true_Y/predict_Y
    
# Fully connected layer 
class FClayer:
    def __init__(self, input_dim, output_dim, batch_size):
        self.input  = None
        self.output = None
        self.w = np.random.rand(input_dim, output_dim) - 0.5
        self.b = np.random.rand(1, output_dim) - 0.5
        self.batch_size = batch_size
        self.name = "FC layer"
        self.error_dw = None
        self.error_db = None
    def forwardPropagation(self, input_data):
        self.input  = input_data
        self.output = np.dot(self.input, self.w) + self.b
        return self.output
    def backPropagation(self, error_dy):
        self.error_dw = np.dot(self.input.T,error_dy)
        self.error_db = error_dy
        error_dx = np.dot(error_dy,self.w.T)
        return error_dx
    def getErrorD(self):
        return self.error_dw, self.error_db
    def updateWeights(self, learning_rate, new_dw, new_db):
        self.w -= learning_rate/self.batch_size*new_dw
        self.b -= learning_rate/self.batch_size*new_db
    def getName(self):
        return self.name

class SoftmaxLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.name = "SM layer"
        self.error_dw = None
        self.error_db = None
    def forwardPropagation(self, X):
        self.X = X
        shiftx = X - np.max(X)
        exps = np.exp(shiftx)
        self.Y = exps / np.sum(exps)
        return self.Y
    def backPropagation(self, output_error):
        out = np.tile(self.Y.T, self.input_size)
        return self.Y * np.dot(output_error, np.identity(self.input_size) - out)
    def getErrorD(self):
        return 0
    def updateWeights(self, learning_rate, new_dw, new_db):
        return 0
    def getName(self):
        return self.name

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
    def backPropagation(self, error_dy):
        return self.activation_prime(self.input)*error_dy
    def getErrorD(self):
        return 0
    def updateWeights(self, learning_rate, new_dw, new_db):
        return 0
    def getName(self):
        return self.name
    
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
                all_w, all_b = [], []
                for b in range(batch_size):
                    input = np.expand_dims(train_data[batch*batch_size+b], axis=0)
                    for layer in self.layers:
                        input = layer.forwardPropagation(input)
                    output_error += self.loss(train_label[batch*batch_size+b], input)
                    # input = np.expand_dims(softmax(input), axis=0)

                    # print('input', input, input.shape)
                    error = self.loss_prime(train_label[batch*batch_size+b], input)

                    # print('error', error, error.shape)
                    temp_w, temp_b = [],[]
                    for layer in reversed(self.layers):
                        error = layer.backPropagation(error)
                        if layer.getErrorD() != 0:
                            w_n, b_n = layer.getErrorD()
                            temp_w.append(w_n)
                            temp_b.append(b_n)
                    all_w.append(temp_w)
                    all_b.append(temp_b)
                    # print('w:',self.layers[0].w)
                # print('====== update w b ========')
                for n_set in range(1,len(all_w)):
                    for bp_w in range(len(all_w[0])):
                        all_w[0][bp_w] += all_w[n_set][bp_w]
                        all_b[0][bp_w] += all_b[n_set][bp_w]
                # print('ww',all_w[0])
                idx = 0
                for layer in reversed(self.layers):
                    if layer.getName() == 'FC layer':
                        # print(layer.w.shape, all_w[0][idx].shape)
                        layer.updateWeights(learning_rate, all_w[0][idx], all_b[0][idx])
                        idx += 1
                # print('upd w -',  all_w[0][-1])
                # print('==========================')
                # print('new w :',  self.layers[0].w)
                # print('****************************')

            print('epoch %d loss : %f'   % (i, output_error))
            self.training_curve_data.append([i, output_error])

        print('Training parameters:\n epochs = %d learning rate = %f' % (epochs, learning_rate))

    # def predictMSE(self, test_data, test_label):
    #     samples, _ = test_data.shape
    #     prediction = np.zeros((samples, 1))
    #     for sample in range(samples):
    #         input = test_data[sample]
    #         for layer in self.layers:
    #             input = layer.forwardPropagation(input)
    #         print(input)
    #         prediction[sample] = np.round(input)
    #     print('pre  ',prediction[80:150])
    #     print('label', test_label)

    #     correct = 0
    #     for sample in range(samples):
    #         if np.array_equal(prediction[sample],test_label[sample]):
    #             correct += 1
    #     print("accuracy: ", correct/samples)
        
    def predict(self, test_data, test_label):
        samples, _ = test_data.shape
        prediction = np.zeros((samples, 3))
        for sample in range(samples):
            input = test_data[sample]
            for layer in self.layers:
                input = layer.forwardPropagation(input)
            input = np.expand_dims(softmax(input), axis=0)
            # print('=================')
            # print(input)
            pos = np.argmax(input, axis=1)#????????
            # print(pos)
            prediction[sample, pos[0]] = 1
            # print('pred', prediction[sample])
        print('pre  ',prediction)
        print('label', test_label)

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



