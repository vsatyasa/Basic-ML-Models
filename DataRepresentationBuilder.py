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
        print(self.dictionary)
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