import os
import sys
import pandas as pd
import snscrape.modules.twitter as sntwitter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

sys.path.append(os.path.join(os.path.dirname(__file__), "..."))
from utils import helper
nltk.download('punkt')
nltk.download('stopwords')
stopwords_ = stopwords.words("english")
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')

def scrap_raw_tweets_from_web(**kwargs) -> None:
    tweets = list()
    try:
        for index, tweet in enumerate(sntwitter.TwitterSearchScraper(kwargs["search_query"]).get_items()):
            if index != kwargs["tweet_limit"]:
                tweets.append([tweet.date, tweet.id, tweet.lang,
                                tweet.user.username, tweet.content])

        raw_tweets_dataframe = pd.DataFrame(
                                            tweets,
                                            columns = [
                                                        'datetime', 'id',
                                                        'lang', 'username',
                                                        'raw_tweets'
                                                    ]
                                            )
        
        raw_tweets_dataframe.to_parquet(kwargs["raw_file_name"],
                                        index = False, engine = "pyarrow")
        kwargs["s3_hook"].load_file(
                                    filename = kwargs["raw_file_name"],
                                    key = kwargs["raw_file_name"],
                                    bucket_name = kwargs["bucket_name"]
                                )

    except Exception as exc:
        raise Exception("Something went wrong with the tweet scraping task. Please check them!!") from exc

def add_sentiment_labels_to_tweets(**kwargs) -> None:
    dataframe = pd.read_parquet(
                                path = f"{kwargs['temp_data_path']}/{kwargs['raw_file_name']}",
                                engine = "pyarrow"
                            )
    dataframe_en = dataframe[dataframe['lang'] == "en"]
    dataframe_en["cleaned_tweets"] = dataframe_en["raw_tweets"].apply(
                                                                lambda text: helper.remove_noise(text)
                                                                )
    dataframe_en["polarity"] = dataframe_en["cleaned_tweets"].apply(
                                                                lambda text: helper.calculate_polarity(text)
                                                                )
    dataframe_en["sentiment"] = dataframe_en["polarity"].apply(
                                                                lambda score: helper.assign_sentiment_labels(score)
                                                            )

    dataframe_en.to_parquet(kwargs["labelled_file_name"],
                            index = True, engine = "pyarrow")
    kwargs["s3_hook"].load_file(
                                filename = kwargs["labelled_file_name"],
                                key = kwargs["labelled_file_name"],
                                bucket_name = kwargs["bucket_name"]
                            )

def preprocess_tweets(**kwargs) -> None:
    dataframe = pd.read_parquet(path = f"{kwargs['temp_data_path']}/{kwargs['labelled_file_name']}",
                                engine = "pyarrow")
    dataframe = dataframe.iloc[: , 1:]
    dataframe['cleaned_tweets'] = dataframe['cleaned_tweets'].astype(str).str.lower()
    dataframe['tokenized_tweets'] = dataframe["cleaned_tweets"].apply(word_tokenize)

    dataframe['tokenized_tweets'] = dataframe['tokenized_tweets'].apply(
                                                                lambda tokens: helper.remove_stopwords(tokens, stopwords_)
                                                                )
    dataframe = helper.remove_less_frequent_words(dataframe)

    wordnet_lem = WordNetLemmatizer()
    dataframe['lemmatized_tweets'] = dataframe['tokenized_strings'].apply(lambda tweet: " ".join([
                                                                            wordnet_lem.lemmatize(word)
                                                                            for word in tweet.split()]))

    porter_stemmer = PorterStemmer()
    dataframe['processed_tweets'] = dataframe['lemmatized_tweets'].apply(lambda tweet: " ".join([
                                                                            porter_stemmer.stem(word)
                                                                            for word in tweet.split()]))

    dataframe = dataframe.reindex(columns = [col for col in dataframe.columns if col != 'sentiment'] + ['sentiment'])

    dataframe['labels'] = dataframe['sentiment'].map(
                                                    {
                                                        "neutral": 0,
                                                        "negative": 1,
                                                        "positive": 2
                                                    }
                                                )
    print(dataframe.shape, dataframe.columns)

    dataframe.to_parquet(kwargs["preprocessed_file_name"],
                        index = False, engine = "pyarrow")
    kwargs["s3_hook"].load_file(
                            filename = kwargs["preprocessed_file_name"],
                            key = kwargs["preprocessed_file_name"],
                            bucket_name = kwargs["bucket_name"]
                        )