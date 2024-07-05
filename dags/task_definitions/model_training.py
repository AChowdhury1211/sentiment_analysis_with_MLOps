import os
import sys
import pandas as pd
from tqdm.auto import tqdm
from dataclasses import dataclass
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import to_categorical
from keras import losses, optimizers, metrics
from transformers import BertTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), "..."))
from utils.helper import load_dataframe
from utils.prepare_data import Dataset
from utils.model import BiLSTM_Model
from utils.helper import Config
from utils.experiment_tracking import MLFlowTracker

config = Config()

@dataclass
class Train_parameters:
        size: int
        num_classes: int
        embedding_dim: int
        sequence_length: int
        num_epochs: int
        learning_rate: float

@dataclass
class Model_tracking_parameters:
        
        experiment_name: str
        mlflow_tracking_uri: str
        run_name: str
        experiment: bool

class Training:
        def __init__(self, training_args: Train_parameters,
                    model_tracking_args: Model_tracking_parameters
                ):

                
                self.training_args = training_args
                self.model_tracking_args = model_tracking_args

        def check_and_set_gpu(self) -> tf.config.LogicalDevice:
                
                try:
                        available_gpu_devices = tf.config.experimental.list_physical_devices("GPU")
                        if len(available_gpu_devices) > 0:
                                # Since the system has only one GPU, setting it to the first GPU
                                tf.config.set_visible_devices(available_gpu_devices[0], "GPU")
                                # Allocating GPU memory based on the runtime
                                tf.config.experimental.set_memory_growth(available_gpu_devices[0], True)
                                logical_gpu = tf.config.list_logical_devices("GPU")

                except Exception as exc:
                        raise RuntimeError("Runtime failed in GPU setting. Please check and try again!!") from exc

                return logical_gpu

        def train(self) -> None:
                
               
                gpu = self.check_and_set_gpu()
                assert len(gpu) > 0

                tracker = MLFlowTracker(experiment_name = self.model_tracking_args.experiment_name,
                                        tracking_uri = self.model_tracking_args.mlflow_tracking_uri,
                                        run_name = self.model_tracking_args.run_name,
                                        experiment = self.model_tracking_args.experiment)
                tracker.log()
                
                dataframe: pd.DataFrame = load_dataframe("./preprocessed_tweets.parquet")
                df = dataframe[['cleaned_tweets','labels']].iloc[0:35000].copy()
                train_dataframe, test_dataframe = train_test_split(df, test_size = 0.25,
                                                                random_state = 42,
                                                                stratify = df['labels'])
                train_dataframe.dropna(inplace = True)
                test_dataframe.dropna(inplace = True)
                
                y_train = to_categorical(train_dataframe['labels'], num_classes = self.training_args.num_classes)
                y_test = to_categorical(test_dataframe['labels'], num_classes = self.training_args.num_classes)

                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                train_dataset: tf.data.Dataset.zip = Dataset(tokenizer = tokenizer, dataframe = train_dataframe,
                                                            labels = y_train, batch_size = self.training_args.batch_size,
                                                            max_length = self.training_args.sequence_length,
                                                            train = True).encode_bert_tokens_to_tf_dataset()

                test_dataset: tf.data.Dataset.zip = Dataset(tokenizer = tokenizer, dataframe = test_dataframe,
                                                           labels = y_test, batch_size = self.training_args.batch_size,
                                                           max_length = self.training_args.sequence_length,
                                                           train = True).encode_bert_tokens_to_tf_dataset()
                                        
                model: Sequential = BiLSTM_Model(
                                tokenizer.vocab_size,
                                self.training_args.num_classes,
                                self.training_args.embedding_dim,
                                self.training_args.sequence_length).create_model()
                
                print("Training started.....")
                model.compile(
                             loss = losses.CategoricalCrossentropy(),
                             optimizer = optimizers.Adam(
                                                        learning_rate = self.training_args.learning_rate,
                                                        epsilon=1e-08),
                             metrics = [metrics.CategoricalAccuracy('accuracy')]
                        )
                
                model.fit(
                        train_dataset,
                        validation_data = test_dataset,
                        epochs = self.training_args.num_epochs,
                        batch_size = self.training_args.batch_size
                        )

                tracker.end()

def main() -> None:
        training_parameters_ = Train_parameters(
                                        config["train-parameters"]["batch_size"],
                                        config["train-parameters"]["num_classes"],
                                        config["train-parameters"]["embedding_dim"],
                                        config["train-parameters"]["sequence_length"],
                                        config["train-parameters"]["num_epochs"],
                                        config["train-parameters"]["learning_rate"],
                                        )

        model_tracking_parameters_ = Model_tracking_parameters(
                                        config["model-tracking"]["experiment_name"],
                                        config["model-tracking"]["mlflow_tracking_uri"],
                                        config["model-tracking"]["run_name"],
                                        config["model-tracking"]["experiment"]
                                        )
        
        model_training_ = Training(
                                training_parameters_,
                                model_tracking_parameters_
                                )

        model_training_.train()

if __name__ == "__main__":
    main()