import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


class TfidfBuilder:
    def __init__(self, documents):
        self.documents = documents
        self.stop_words = set(stopwords.words('english'))
        self.porter = PorterStemmer()

    def build(self):
        tfidf = {}
        for document in self.documents:
            for word in self._tokenize(document):
                if word not in tfidf:
                    tfidf[word] = 0
                tfidf[word] += 1

        return tfidf

    def _tokenize(self, document):
        tokens = word_tokenize(document)
        tokens = [t.lower() for t in tokens]
        tokens = [t for t in tokens if t not in string.punctuation]
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.porter.stem(t) for t in tokens]
        return tokens