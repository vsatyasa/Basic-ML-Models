
import numpy as np
import math
from DataRepresentationBuilder import DataRepresentationBuilder

class MultiLayerNN():
    
    
    class Layer():
        
        def __init__(self, input_size, output_size) -> None:
            
            limit = 1 / math.sqrt(input_size)
            self.weights = np.random.uniform(-1 * limit, limit, (input_size, output_size))
            # self.weights = np.random.rand(input_size, output_size)
            self.bias = np.random.rand(1, output_size)

            # store the last inputs and outputs
            self.last_input = None
            self.last_output = None
        
        def forward(self, X):
            return np.dot(X, self.weights) + self.bias

        def activation(self, X):
            return 1 / (1 + np.exp(-X))

        def forward_with_activation(self, X):
            forward_out = self.forward(X)
            out = self.activation(forward_out)
            self.last_input = X
            self.last_output = out
            return out, forward_out
            

        

    def __init__(
        self, 
        X_train, 
        y_train, 
        epochs,
        learning_rate,
        hidden_layer_size,
        one_hot_classes_map,
        batch_size = None) -> None:
        
        self.X_train = X_train
        self.y_train = y_train
        self.one_hot_classes_map = one_hot_classes_map
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        
        ## init the network
        ## init weights and bias         
        self.layers = []
        self.layers.append(self.Layer(X_train.shape[1], hidden_layer_size))
        self.layers.append(self.Layer(hidden_layer_size, y_train.shape[1]))
    
    def forward_propagation(self, X):
        f1, z1 = self.layers[0].forward_with_activation(X)
        f2, z2 = self.layers[1].forward_with_activation(f1)
        return f2

    def softmax(self, x):  
        max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(x - max) #subtracts each row with its max value
        sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / sum 
        return f_x

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backward_propagation(self, y, z):
        last_layer_error = z - y
        last_layer_delta = last_layer_error * self.sigmoid_derivative(z) # consider the loss function as log loss
        self.layers[1].weights -= self.learning_rate * np.dot(self.layers[1].last_input.T, last_layer_delta)
        self.layers[1].bias -= self.learning_rate * np.sum(last_layer_delta, axis=0, keepdims=True)
        
        first_layer_error = np.dot(last_layer_delta, self.layers[1].weights.T)
        first_layer_delta = first_layer_error * self.sigmoid_derivative(self.layers[0].last_output)
        self.layers[0].weights -= self.learning_rate * np.dot(self.layers[0].last_input.T, first_layer_delta)
        self.layers[0].bias -= self.learning_rate * np.sum(first_layer_delta, axis=0, keepdims=True)
        
    def predict(self, X):
        last_layer_output = self.forward_propagation(X)
        predicted_class = np.argmax(last_layer_output, axis=1)
        one_hot_predicted_class = np.zeros(last_layer_output.shape[1])
        one_hot_predicted_class[predicted_class] = 1
        return one_hot_predicted_class
    
    def test_and_get_accuracy(self, X_test, y_test):
        correcect_predictions = 0
        for i in range(len(X_test)):
            x = X_test[i]
            y = y_test[i]
            y1 = self.predict(x)
            if (y == y1).all():
                correcect_predictions += 1
        return (correcect_predictions / len(X_test)) * 100
            
    def train(self):
        ## repeat the training for the number of epochs
        for i in range(self.epochs):
            out = self.forward_propagation(self.X_train)
            self.backward_propagation(self.y_train,  out)


class CrossValidationTrain():
    
    learning_rate = 0.001
    epochs = 5000
    
    def __init__(self, k, X_train, Y_train, vocab_map) -> None:
        self.k = k
        self.vocab_map = vocab_map
        
        self.x_folds, self.y_folds = self.divide_to_k_folds(X_train, Y_train)
    
    def divide_to_k_folds(self, X_train, Y_train):
        x_folds = []
        y_folds = []
        
        for i in range(self.k):
            x_folds.append([])
            y_folds.append([])
        
        indexes = []
        for i in range(len(X_train)):
            indexes.append(i)
        np.random.shuffle(indexes)
        
        k_fold_indexes = np.array_split(indexes, self.k)
        
        for i  in range(len(k_fold_indexes)):
            fold = k_fold_indexes[i]
            for j in range(len(fold)):
                idx = fold[j]
                x_folds[i].append(X_train[idx])
                y_folds[i].append(Y_train[idx])
        
        return x_folds, y_folds

    def train_and_test_set(self, i):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        
        for j in range(self.k):
            
            if j == i:
                test_x = self.x_folds[j]
                test_y = self.y_folds[j]
            else:
                train_x.extend(self.x_folds[j])
                train_y.extend(self.y_folds[j])
        

        return train_x, train_y, test_x, test_y
        
    def train_nn_and_record_accuracy(self):
        
        accuracies = []
        
        for i in range(self.k):
            X_train, Y_train, X_test, Y_test = self.train_and_test_set(i)
            
            print(len(X_train), len(Y_train), len(X_test), len(Y_test))
            lr = MultiLayerNN(
                np.array(X_train),
                np.array(Y_train),
                self.epochs,
                self.learning_rate,
                6, # hidden layer size
                vocab_map
            )
            lr.train()
            
            accuracies.append(lr.test_and_get_accuracy(X_test, Y_test)) 
        print(accuracies)   
        print(np.average(accuracies))
        



dataRepresentationBuilder = DataRepresentationBuilder("train.csv")
X_train, Y_train, vocab_map = dataRepresentationBuilder.get_training_set_and_vocab()



# load iris dataset
# f = open('iris.data.csv', 'r')
# X_train = list()
# Y_train = []

# y_set = set()
# for line in f:
#     line = line.strip()
#     line = line.split(',')
#     X_train.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])
#     Y_train.append(line[4])
#     y_set.add(line[4])
# vocab_map = {}
# for i in range(len(y_set)):
#     one_hot = [0] * len(y_set)
#     one_hot[i] = 1
#     vocab_map[list(y_set)[i]] = one_hot
# Y_train = [vocab_map[y] for y in Y_train]



cross_validation_train = CrossValidationTrain(5, X_train, Y_train, vocab_map)
cross_validation_train.train_nn_and_record_accuracy()