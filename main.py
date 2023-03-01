import numpy as np
import math
import preprocessor as p
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim import models
import nltk
import numpy as np
from gensim.parsing.porter import PorterStemmer


# nltk.download('wordnet')
## save the network call delay
## by init the stop words manually
NLTK_STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
                   "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
                   'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                   'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                   'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
                   'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
                   'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 
                   'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
                   'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 
                   'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
                   'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
                   "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
                   'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
                   'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


## Class for Loading data
## and building the tf-idf model
class DataRepresentationBuilder():
    
    train_tweets = []
    
    def __init__(self, tweets_train_url) -> None:
        
        ## stop words
        self.stop_words = NLTK_STOP_WORDS
        
        ## stemmer
        self.stemmer = PorterStemmer()
        
        ## tf idf model building
        self.tf_idf_model = None
        self.tweets_train_url = tweets_train_url
        self.train_tweets=[]
        self.train_emotions_labels = []
        self.train_emotions_vocab = {}
        self.read_tweets(tweets_train_url)
        self.build_tf_idf_model()

    def get_tf_idf_model(self):
        return self.tf_idf_model

    def read_tweets(self, tweets_train):
        f = open(tweets_train, "r")
        tweets = f.readlines()
        tweets = tweets[1:]
        for tweet in tweets:
            Id, tweet, emotion = tweet.split(',', 2)
            # tweet = tweet.strip("\n")
            self.train_tweets.append(tweet)
            self.train_emotions_labels.append(emotion.strip("\n"))

    def build_tf_idf_model(self):
        self._clean_data()
        self.build_vocab()
        self._init_idf_model()

    def _clean_data(self):
        
        cleaned_tweets = []
        # clean the stop words
        for tweet in self.train_tweets:
            tweet_tokens = tweet.split(' ')
            cleaned_tweet = []
            for word in tweet_tokens:
                if word not in self.stop_words:
                    cleaned_tweet.append(self.stemmer.stem(word))
            cleaned_tweets.append(cleaned_tweet)
        self._clean_train_tweets = cleaned_tweets


    def build_vocab(self):
        self.vocab_map = {}
        vocab_index = 0
        
        for tweet in self._clean_train_tweets:
            for word in tweet:
                if word not in self.vocab_map:
                    self.vocab_map[word] = vocab_index
                    vocab_index += 1
        
        emotion_set = set(self.train_emotions_labels)
        emotion_set = list(emotion_set)
        for i in range(len(emotion_set)):
            one_hot = np.zeros(len(emotion_set))
            one_hot[i] = 1
            self.train_emotions_vocab[emotion_set[i]] = one_hot
        
        self.vovab_size = len(self.vocab_map)        

    def _init_idf_model(self):
        # doc_tokenized = [simple_preprocess(doc) for doc in self._clean_train_tweets]
        self.dictionary = corpora.Dictionary()
        self.BoW_corpus = [self.dictionary.doc2bow(doc, allow_update=True) for doc in self._clean_train_tweets]
        self.tf_idf_model = models.TfidfModel(self.BoW_corpus, smartirs='ntc')
        
        
    def get_tf_idf_vectors(self):
        
        feature_vectors = []
        for bow in self.BoW_corpus:
            vec = self.tf_idf_model[bow]
            feature_vector = np.zeros(self.vovab_size)
            
            for word_index, tf_idf in vec:
                feature_vector[word_index - 1] = tf_idf
            
            # print(feature_vector)
            feature_vectors.append(feature_vector)
        
        return feature_vectors

    def get_emotions_one_hot(self):
        emotions_one_hot = []
        
        for emotion in self.train_emotions_labels:
            emotions_one_hot.append(self.train_emotions_vocab[emotion])
        
        return emotions_one_hot
    
    def get_test_set_representation(self):
        f = open("test.csv", "r")
        
        id_tweet_vec = []
        data = f.readlines()
        
        for line in data[1:]:
            id, tweet = line.strip("\n").split(',', 1)

            cleaned_tweet = []
            for word in tweet.split(' '):
                if word not in self.stop_words:
                    cleaned_tweet.append(self.stemmer.stem(word))
            
            feature_vector = np.zeros(self.vovab_size)
            rep  = self.tf_idf_model[self.dictionary.doc2bow(cleaned_tweet)]
            for word_index, tf_idf in rep:
                feature_vector[word_index - 1] = tf_idf
            
            id_tweet_vec.append((id, tweet, feature_vector))      
            
        return id_tweet_vec  
    
    def get_training_set_and_vocab(self):
        return self.get_tf_idf_vectors(), self.get_emotions_one_hot(), self.train_emotions_vocab


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
    
    def predict_and_get_accuracy_and_confusion_matrix(self, X_test, y_test):
        correct_predictions = 0
        confusion_matrix = np.zeros((len(self.one_hot_classes_map), len(self.one_hot_classes_map)))
        for i in range(len(X_test)):
            x = X_test[i]
            y = y_test[i]
            y1 = self.predict_one_hot_class(x)
            if (y == y1).all():
                correct_predictions += 1
            
            pred_class = np.argmax(y1)
            actual_class = np.argmax(y)
            confusion_matrix[actual_class][pred_class] += 1
        print(confusion_matrix)
        
        return (correct_predictions / len(X_test)) * 100, confusion_matrix
    
    
    def predict_test_and_train(self, X_test, y_test):
        return self.predict_and_get_accuracy(self.X_train, self.y_train), self.predict_and_get_accuracy(X_test, y_test)
    
    def get_emotion_class(self, one_hot_class):
        for key, value in self.one_hot_classes_map.items():
            if (value == one_hot_class).all():
                return key
        return self.one_hot_classes_map[one_hot_class]    
 
    def predict_and_print_file(self, id_tweet_vec, file_name):
        f = open(file_name, "w")
        f.write("id,tweet,emotions\n")
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


class MultiLayerNN():
    
    ## Layer Implementation 
    # reference: https://blog.zhaytam.com/2018/08/15/implement-neural-network-backpropagation/
    class Layer():
        
        def __init__(self, input_size, output_size):
            
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
        return self.softmax(z2)

    def softmax(self, x):  
        max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(x - max) #subtracts each row with its max value
        sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / sum 
        return f_x

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    
    ## backpropagation layers
    ## reference: https://www.cristiandima.com/neural-networks-from-scratch-in-python
    def backward_propagation(self, y, z):
        last_layer_error = z - y
        last_layer_delta = last_layer_error # consider the loss function as log loss
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

    def predict_test_and_train(self, X_test, y_test):
        return self.test_and_get_accuracy(self.X_train, self.y_train), self.test_and_get_accuracy(X_test, y_test)

    def get_emotion_class(self, one_hot_class):
        for key, value in self.one_hot_classes_map.items():
            if (value == one_hot_class).all():
                return key
        return self.one_hot_classes_map[one_hot_class]    
 
    def predict_and_print_file(self, id_tweet_vec, file_name):
        
        f = open(file_name, "w")
        f.write("id,tweet,emotions\n")
        for id, tweet, vec in id_tweet_vec:
            y_pred = self.predict(vec)
            pred_emotion = self.get_emotion_class(y_pred)
            f.write(str(id) + "," + str(tweet) + "," + str(pred_emotion) + "\n")
            
    def train(self):
        ## repeat the training for the number of epochs
        for i in range(self.epochs):
            out = self.forward_propagation(self.X_train)
            self.backward_propagation(self.y_train,  out)


if __name__ == '__main__':
    
    ## load the data required for training and testing
    dataRepresentationBuilder = DataRepresentationBuilder("train.csv")
    X_train, Y_train, vocab_map = dataRepresentationBuilder.get_training_set_and_vocab()
    id_tweet_vec = dataRepresentationBuilder.get_test_set_representation()
    
    print ("..................Beginning of Logistic Regression................")
    
    learning_rate = 0.0008
    epochs = 2000
    lr = LiogisticRegression(
        learning_rate,
        epochs,
        np.array(X_train),
        np.array(Y_train),
        vocab_map
    )
    lr.train()
    lr.predict_and_print_file(id_tweet_vec, "test_lr.csv")    
    
    print ("..................End of Logistic Regression................")

    print("\n------------------------------------------------\n")

    print ("..................Beginning of Neural Network................")
    learning_rate = 0.001
    epochs = 2000
    lr = MultiLayerNN(
        np.array(X_train),
        np.array(Y_train),
        epochs,
        learning_rate,
        10, # hidden layer size
        vocab_map
    )
    lr.train()
    lr.predict_and_print_file(id_tweet_vec, "test_nn.csv")


    print ("..................End of Neural Network................")



class CrossValidationTrainLR():
    
    learning_rate = 0.0008
    epochs = 2000
    
    def __init__(self, k, X_train, Y_train, vocab_map, lr) -> None:
        self.k = k
        self.vocab_map = vocab_map
        self.learning_rate = lr
        
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
        training_accuracies = []
        testing_accuraies = []
        
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
            trac, teac = lr.predict_test_and_train(X_test, Y_test)
            # print("Train accuracy: ", trac)
            # print("Test accuracy: ", teac)
            # print("=====================================")         
            testing_accuraies.append(teac)
            training_accuracies.append(trac)
        #     accuracies.append(lr.predict_and_get_accuracy(X_test, Y_test)) 
        print("===================================== \n\n")
        print("Learning Rate: " + str(self.learning_rate))   
        print("Training Acc" + str(np.average(training_accuracies)))
        print("Testing Acc" + str(np.average(testing_accuraies)))
        print("===================================== \n\n")


class CrossValidationTrainNN():
    
    learning_rate = 0.001
    epochs = 5000
    
    def __init__(self, k, X_train, Y_train, vocab_map, epochs) -> None:
        self.k = k
        self.vocab_map = vocab_map
        self.epochs = epochs
        
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
        
        training_accuracies = []
        testing_accuraies = []
        
        for i in range(self.k):
            X_train, Y_train, X_test, Y_test = self.train_and_test_set(i)
            
            print(len(X_train), len(Y_train), len(X_test), len(Y_test))
            lr = MultiLayerNN(
                np.array(X_train),
                np.array(Y_train),
                self.epochs,
                self.learning_rate,
                10, # hidden layer size
                vocab_map
            )
            lr.train()
            trac, teac = lr.predict_test_and_train(X_test, Y_test)
            testing_accuraies.append(teac)
            training_accuracies.append(trac)
            # accuracies.append(lr.test_and_get_accuracy(X_test, Y_test)) 
        print("===================================== \n\n")
        print("Training Acc" + str(np.average(training_accuracies)))
        print("Testing Acc" + str(np.average(testing_accuraies)))
        print("===================================== \n\n")