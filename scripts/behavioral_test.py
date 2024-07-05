import os
import spacy
import numpy as np
import pandas as pd
import tensorflow as tf
from checklist.perturb import Perturb
from keras.models import Sequential
from sklearn.metrics import accuracy_score
nlp = spacy.load('en_core_web_sm')


def min_functionality_test(dataframe: pd.DataFrame) -> pd.DataFrame:
    

    original_text: list = dataframe["sample_text"].tolist()
    true_labels: list = dataframe["labels"].tolist()
    piped_text  = list(nlp.pipe(original_text))

   
    perturbed_data = Perturb.perturb(piped_text, Perturb.add_negation)
    negated_texts: list = [text[1] for text in perturbed_data.data]

    negated_dataframe = pd.DataFrame(
                                    list(zip(negated_texts, true_labels)),
                                    columns = ["negated_text", "labels"]
                                    )

    return negated_dataframe

def invariance_test(text: str) -> str:
    

    text_with_typo = str(Perturb.add_typos(text))
    perturbed_text = Perturb.expand_contractions(text_with_typo)
    return perturbed_text


def run(test_name: str, model: Sequential,
        test_dataset: tf.data.Dataset.zip,
        dataframe: pd.DataFrame) -> float:
    
    try:
        for text, _ in test_dataset.take(1):
            text_ = text.numpy()

    except Exception:
        print(f"Exception occurred when trying to access {test_dataset}. Please check!!")
    
    else:
        predicted_probabilities = model.predict(text_)
        predicted_labels = np.argmax(
                                    np.array(predicted_probabilities),
                                    axis = 1
                                    )

        dataframe["predicted_labels"] = predicted_labels
        dataframe["predicted_probabilities"] = predicted_probabilities.tolist()

        # Save test results as CSv
        dataframe_path = os.path.join(os.getcwd(), "test_results")
        dataframe.to_csv(f"{dataframe_path}/{test_name}_test_results.csv", index = False)

        test_accuracy = accuracy_score(
                                        y_true = dataframe['labels'].tolist(),
                                        y_pred = dataframe['predicted_labels'].tolist()
                                    )

        return test_accuracy