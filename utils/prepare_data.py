
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from transformers import BertTokenizer

@dataclass
class Dataset:
    
    tokenizer: BertTokenizer
    dataframe: pd.DataFrame = field(default_factory = pd.DataFrame())
    labels: np.ndarray = None
    batch_size: int = 64
    max_length: int = 256
    train: bool = False
    col_name: str = "cleaned_tweets"

    @property
    def list_of_texts(self) -> list[str]:
        
        return self.dataframe[self.col_name].tolist()

    @property
    def shuffle_size(self) -> int:
        
        return math.ceil(len(self.list_of_texts) / self.batch_size)

    def encode_bert_tokens_to_tf_dataset(self) -> tf.data.Dataset.zip:
       
        tokenized: BertTokenizer = self.tokenizer(
                                                text = self.list_of_texts,
                                                add_special_tokens = True,
                                                max_length = self.max_length,
                                                padding = "max_length",
                                                return_tensors = "tf",
                                                return_attention_mask = False,
                                                return_token_type_ids = False,
                                                verbose = True
                                            )

        input_ids = tf.data.Dataset.from_tensor_slices(np.array(tokenized['input_ids']))
        labels = tf.data.Dataset.from_tensor_slices(self.labels)
        
        dataset = tf.data.Dataset.zip((input_ids, labels))

        if self.train:
            return dataset.shuffle(self.shuffle_size).batch(self.batch_size)

        return dataset.batch(self.batch_size)