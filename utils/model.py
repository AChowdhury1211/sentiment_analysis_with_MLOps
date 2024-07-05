

from abc import ABC, abstractmethod
from dataclasses import dataclass
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

class Models(ABC):
    
    @abstractmethod
    def define_model(self):
        pass

    @abstractmethod
    def create_model(self):
        pass


@dataclass
class BiLSTM_Model(Models):
   
    vocab_size: int
    num_classes: int
    embedding_dim: int = 64
    input_length: int = 128

    def define_model(self) -> Sequential:
       
        return Sequential(
                    [

                    
                    Embedding(self.vocab_size, self.embedding_dim, input_length = self.input_length),

                    
                    Bidirectional(LSTM(self.embedding_dim, return_sequences=True)),
                    Bidirectional(LSTM(64, return_sequences = True)),
                    Bidirectional(LSTM(32)),
                    
                    
                    Dense(self.embedding_dim, activation = 'relu'),
                    Dense(64, activation = 'relu'),
                    Dropout(0.25),
                    Dense(self.num_classes, activation = 'softmax')
                    ]
                )
        
    def create_model(self) -> Sequential:
        

        model: Sequential = self.define_model()
        model.summary()
        return model