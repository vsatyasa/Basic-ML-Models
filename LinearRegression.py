## Multinomial Logistic Regression
import numpy as np
import math
import pandas as pd
from DataRepresentationBuilder import DataRepresentationBuilder

class LiogisticRegression():
    
    def __init__(self,          
            learning_rate, 
            epochs,
            X_train,
            y_train,
            one_hot_classes_map
        ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.X_train = X_train
        self.y_train = y_train
        self.one_hot_classes_map = one_hot_classes_map

        ## init the weights and bias
        self.seed_weights(self.X_train.shape[1], self.y_train.shape[1])
    
    def seed_weights(self, n_features, n_classes):
        limit = 1 / math.sqrt(n_features)
        self.W = np.random.uniform(-1 * limit, limit, (n_features, n_classes))
        self.b = np.zeros((1, n_classes))

    # def softmax(self, z):   
    #     softmax_out = (np.exp(z) / np.sum(np.exp(z), axis=1)).round(10)
    #     return softmax_out

    def softmax(self, x):  
        max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(x - max) #subtracts each row with its max value
        sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / sum 
        return f_x
    
    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        return self.softmax(z)
    
    def predict_one_hot_class(self, X):
        y_pred = self.predict(X)
        pred_class =  np.argmax(y_pred, axis=1)
        # print(y_pred)
        one_hot_pred_class = np.zeros(y_pred.shape[1])
        one_hot_pred_class[pred_class] = 1
        # print(one_hot_pred_class)
        return one_hot_pred_class
    
    ## useful for cross validation``
    def predict_and_get_accuracy(self, X_test, y_test):
        correct_predictions = 0
        for i in range(len(X_test)):
            x = X_test[i]
            y = y_test[i]
            y1 = self.predict_one_hot_class(x)
            if (y == y1).all():
                correct_predictions += 1
        
        return (correct_predictions / len(X_test)) * 100
    
    def get_emotion_class(self, one_hot_class):
        for key, value in self.one_hot_classes_map.items():
            if (value == one_hot_class).all():
                return key
        return self.one_hot_classes_map[one_hot_class]    
 
    def predict_and_print_file(self, id_tweet_vec, file_name):
        
        f = open(file_name, "w")
        for id, tweet, vec in id_tweet_vec:
            y_pred = self.predict_one_hot_class(vec)
            pred_emotion = self.get_emotion_class(y_pred)
            f.write(str(id) + "," + str(tweet) + "," + str(pred_emotion) + "\n")

    
    ## Reference for update weights: https://www.kaggle.com/code/vitorgamalemos/multinomial-logistic-regression-from-scratch
    def update_weights(self, X, y, y_pred):
        self.W = self.W - self.learning_rate * np.dot(X.T, (y_pred - y)) ## cross check this
        self.b = self.b - self.learning_rate * np.sum(y_pred - y, axis=0) ## cross check this
        
    def forward(self, X):
        z = np.dot(X, self.W) + self.b
        return self.softmax(z)
    
    def train(self):
        for i in range(self.epochs):
            Z = self.forward(self.X_train)
            self.update_weights(self.X_train, self.y_train, Z)
            

class CrossValidationTrain():
    
    learning_rate = 0.1
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
        
    def train_lr_and_record_accuracy(self):
        
        accuracies = []
        
        for i in range(self.k):
            X_train, Y_train, X_test, Y_test = self.train_and_test_set(i)
            
            # print(len(X_train), len(Y_train), len(X_test), len(Y_test))
            lr = LiogisticRegression(
                self.learning_rate,
                self.epochs,
                np.array(X_train),
                np.array(Y_train),
                vocab_map
            )
            lr.train()
            
            accuracies.append(lr.predict_and_get_accuracy(X_test, Y_test))    
        print(np.average(accuracies))
        



dataRepresentationBuilder = DataRepresentationBuilder("train.csv")
X_train, Y_train, vocab_map = dataRepresentationBuilder.get_training_set_and_vocab()
id_tweet_vec = dataRepresentationBuilder.get_test_set_representation()

learning_rate = 0.1
epochs = 5000
lr = LiogisticRegression(
    learning_rate,
    epochs,
    np.array(X_train),
    np.array(Y_train),
    vocab_map
)
lr.train()
lr.predict_and_print_file(id_tweet_vec, "output.csv")
