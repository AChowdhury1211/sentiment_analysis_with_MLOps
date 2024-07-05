
import os
import mlflow
import sys
import pandas as pd
import tensorflow as tf
import behavioral_test
from dataclasses import dataclass, field
from keras.utils import to_categorical
from transformers import BertTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import Config, load_dataframe
from utils.prepare_data import Dataset

config = Config()

@dataclass
class Productionalize:
    
    tracking_uri: str
    test_data: str = "./test_data.parquet"
    client: mlflow.MlflowClient = None
    test_dataframe: pd.DataFrame = None
    model_name: str = ""
    batch_size: int = 64
    sequence_length: int = 256
    num_classes: int = 3
    latest_version: int = 3
    filter_string = "name LIKE 'sentiment%'"

    def __post_init__(self) -> None:
        
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        except ConnectionError:
            print(f"Cannot connect to {self.tracking_uri}. Please check and try again!!!")

        else:
            self.client = mlflow.MlflowClient()
            self.latest_version = self.client.get_latest_versions(name = self.model_name)[0].version
            self.test_dataframe = load_dataframe(self.test_data)

    def get_all_registered_models(self) -> None:
        
        for model in self.client.search_registered_models(filter_string = self.filter_string):
            for model_version in model.latest_versions:
                print(f"name = {model_version.name}, version = {model_version.version}, stage = {model_version.current_stage}, run_id = {model_version.run_id}")

    def load_models(self) -> tf.function:
       

        latest_model: tf.function = mlflow.tensorflow.load_model(
                                                            model_uri = f"models:/{self.model_name}/{self.latest_version}"
                                                            )

        production_model: tf.function = mlflow.tensorflow.load_model(
                                                            model_uri = f"models:/{self.model_name}/production"
                                                            )

        return latest_model, production_model

    def transform_data(self, dataframe: pd.DataFrame,
                      col_name: str = "cleaned_tweets") -> tf.data.Dataset.zip:
       

        y_test = to_categorical(dataframe['labels'], self.num_classes)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = Dataset(tokenizer = tokenizer, dataframe = dataframe,
                          labels = y_test, batch_size = self.batch_size,
                          max_length = self.sequence_length,
                          col_name = col_name).encode_bert_tokens_to_tf_dataset()

        return dataset

    def benchmark_models(self) -> tuple[tuple[float], tuple[float]]:
        

        latest_model, production_model = self.load_models()

        sample_mft_dataframe = load_dataframe("./scripts/test_data/sample_test_data_for_mft.parquet")
        negated_dataframe = behavioral_test.min_functionality_test(sample_mft_dataframe)
        perturbed_dataset_mft = self.transform_data(dataframe = negated_dataframe, col_name = "negated_text")
        accuracy_latest_model_mft = behavioral_test.run(test_name = "MFT_latest", model = latest_model,
                                                        test_dataset = perturbed_dataset_mft, dataframe = negated_dataframe)
        accuracy_production_model_mft = behavioral_test.run(test_name = "MFT_production", model = production_model,
                                                        test_dataset = perturbed_dataset_mft, dataframe = negated_dataframe)

    
        perturbed_dataframe_inv = self.test_dataframe.tail(100)
        perturbed_dataframe_inv["cleaned_tweets"] = perturbed_dataframe_inv["cleaned_tweets"].apply(
                                                                            lambda text: behavioral_test.invariance_test(text)
                                                                            )
        perturbed_dataset_inv = self.transform_data(dataframe = perturbed_dataframe_inv)
        accuracy_latest_model_inv = behavioral_test.run(test_name = "Invariance_latest", model = latest_model,
                                                        test_dataset = perturbed_dataset_inv, dataframe = perturbed_dataframe_inv)
        accuracy_production_model_inv = behavioral_test.run(test_name = "Invariance_production", model = production_model,
                                                        test_dataset = perturbed_dataset_inv, dataframe = perturbed_dataframe_inv)

   
        test_dataset = self.transform_data(dataframe = self.test_dataframe)
        latest_model_score = latest_model.evaluate(test_dataset)
        production_model_score = production_model.evaluate(test_dataset)


        latest_model_accuracies = (accuracy_latest_model_mft, accuracy_latest_model_inv, latest_model_score[1])
        production_model_accuracies = (accuracy_production_model_mft, accuracy_production_model_inv, production_model_score[1])

        return latest_model_accuracies, production_model_accuracies

    def push_new_model_to_production(self, latest_model_accuracies: tuple[float],
                                    production_model_accuracies: tuple[float]) -> bool:
        
        print(f"Latest model accuracies: {latest_model_accuracies},\nProduction model accuracies: {production_model_accuracies}")

        if latest_model_accuracies > production_model_accuracies:
            self.client.transition_model_version_stage(
                                                    name = self.model_name,
                                                    version = self.latest_version,
                                                    stage = "Production")

            print("Transitioned latest model to production!!")
            success = True

        else:
            print("Cannot transition the model stage. Latest model cannot outperform production model in all conducted tests!!!")
            success = False

        return success

def main() -> None:
    productionalize_ = Productionalize(tracking_uri = config["model-tracking"]["mlflow_tracking_uri"],
                                    test_data = config["files"]["test_data"],
                                    model_name = config["model-registry"]["model_name"],
                                    batch_size = config["train-parameters"]["batch_size"],
                                    sequence_length = config["train-parameters"]["sequence_length"]
                                    )

    accuracy_latest_model, accuracy_production_model = productionalize_.benchmark_models()

    success_ = productionalize_.push_new_model_to_production(accuracy_latest_model, accuracy_production_model)

    if success_:
        productionalize_.get_all_registered_models()

if __name__ == "__main__":
    main()