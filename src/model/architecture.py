import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from gensim.models import KeyedVectors

class SentimentLSTMModel:
    def __init__(self, vocab_size, embedding_dim, lstm_units, dense_units, 
                 dropout_rate, num_classes, max_length):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.max_length = max_length
        self.model = None
    
    def build_model(self, word_index=None, word2vec_path=None):
        self.model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, 
                     input_length=self.max_length,
                     trainable=True),
            LSTM(self.lstm_units),
            Dropout(self.dropout_rate),
            Dense(self.dense_units, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        if word_index and word2vec_path:
            self._load_word2vec_weights(word_index, word2vec_path)
        
        return self.model
    
    def _load_word2vec_weights(self, word_index, word2vec_path):
        try:
            word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
            embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
            
            for word, i in word_index.items():
                if i >= self.vocab_size:
                    continue
                try:
                    embedding_matrix[i] = word2vec[word]
                except KeyError:
                    embedding_matrix[i] = np.random.normal(0, 0.1, self.embedding_dim)
            
            self.model.layers[0].set_weights([embedding_matrix])
        except:
            pass
    
    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def save_model(self, path):
        self.model.save(path)
    
    def load_model(self, path):
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
