import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextTokenizer:
    def __init__(self, vocab_size, max_length):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = KerasTokenizer(num_words=vocab_size, oov_token='<UNK>')
    
    def tokenize_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in self.stop_words]
        return tokens
    
    def fit_on_texts(self, texts):
        tokenized = [' '.join(self.tokenize_text(t)) for t in texts]
        self.tokenizer.fit_on_texts(tokenized)
    
    def texts_to_sequences(self, texts):
        tokenized = [' '.join(self.tokenize_text(t)) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(tokenized)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        return padded
    
    def save_tokenizer(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def load_tokenizer(self, path):
        with open(path, 'rb') as f:
            self.tokenizer = pickle.load(f)
