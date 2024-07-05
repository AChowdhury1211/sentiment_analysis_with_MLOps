
import re
import os
import nltk
import pandas as pd
from textblob import TextBlob
from nltk.probability import FreqDist
import tomli as tomlib
from typing import Any
from dataclasses import dataclass
from airflow import settings
from airflow.exceptions import AirflowFailException
from airflow.models.connection import Connection

class Config:
    
    def __new__(cls) -> dict[str, Any]:
        

        with open("./config/config.toml", mode="rb") as config_file:
            config = tomlib.load(config_file)
        return config

def load_dataframe(file_path: str) -> pd.DataFrame:
   
    this_dir = os.getcwd()
    dataframe_path = os.path.join(this_dir, file_path)
    dataframe = pd.read_parquet(path = dataframe_path, engine = "pyarrow")
    return dataframe

@dataclass
class Connections:
    
    new_connection: Connection

    def create_connections(self) -> bool:
        
        try:
            session = settings.Session()
            connection_name = session.query(Connection).filter(
                                                        Connection.conn_id == self.new_connection.conn_id
                                                        ).first()

            if str(connection_name) != str(self.new_connection.conn_id):
                session.add(self.new_connection)
                session.commit()

        except Exception as exc:
            raise AirflowFailException( f"Error when creating new connection:{exc}") from exc

        else:
            return True
        
        finally:
            session.close()

def remove_noise(text: str) -> str:
    

    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
            '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', text)
    text = re.sub("(@[A-Za-z0-9_]+)","", text)
    text = re.sub('\n',' ', text)
    text = re.sub('#','', text)
    
    return text

def calculate_polarity(text: str) -> float:
    
    return TextBlob(text).sentiment.polarity

def remove_stopwords(tokens: list[str],
                    stopwords_: nltk.corpus.stopwords) -> list[str]:
    
    return [token for token in tokens if token not in stopwords_]

def remove_less_frequent_words(dataframe) -> pd.DataFrame:
    

    dataframe['tokenized_strings'] = dataframe['tokenized_tweets'].apply(
                                                                    lambda tokens: ' '.join(
                                                                                    [token for token in tokens if len(token) > 2]
                                                                                    )
                                                                        )
    tokenized_words = nltk.tokenize.word_tokenize(' '.join(
                                                            [word
                                                            for word in dataframe['tokenized_strings']
                                                            ]
                                                        )
                                                )
    frequency_distribution = FreqDist(tokenized_words)
    dataframe['tokenized_strings'] = dataframe['tokenized_tweets'].apply(
                                                                        lambda tweets: ' '.join(
                                                                                [tweet for tweet in tweets
                                                                                if frequency_distribution[tweet] > 2
                                                                                ]
                                                                            )
                                                                        )
    return dataframe

def assign_sentiment_labels(score: float) -> str:
    
    if score > 0.25:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"