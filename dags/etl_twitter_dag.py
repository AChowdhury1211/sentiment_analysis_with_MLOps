
import os
import json
import sys
from datetime import datetime
from airflow.decorators import task, dag
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from snowflake.connector.pandas_tools import write_pandas
from airflow.models.connection import Connection
from task_definitions.etl_task_definitions import scrap_raw_tweets_from_web, preprocess_tweets
from task_definitions.etl_task_definitions import add_sentiment_labels_to_tweets

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import Config, Connections
from utils.helper import load_dataframe


config = Config()

@dag(dag_id = "etl", start_date = datetime(2023,1,1), schedule_interval = "@monthly", catchup = False)
def twitter_data_pipeline_dag_etl() -> None:
   

    @task(task_id = "configure_connections")
    def set_connections() -> None:
      
        aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        aws_region_name = os.environ["REGION"]
        s3_credentials = json.dumps(
                                dict(
                                    aws_access_key_id = aws_access_key_id,
                                    aws_secret_access_key = aws_secret_access_key,
                                    aws_region_name = aws_region_name,
                                    )
                                )

        s3_connection = Connection(conn_id = "s3_connection",
                                   conn_type = "S3",
                                   extra = s3_credentials
                                  )
        s3_conn_response = Connections(s3_connection).create_connections()


        login = os.environ["LOGIN"]
        password = os.environ["PASSWORD"]
        host_name = os.environ["HOST"]

        snowflake_connection = Connection(conn_id = "snowflake_conn",
                                          conn_type = "Snowflake",
                                          host = host_name,
                                          login = login,
                                          password = password
                                        )

        snowflake_conn_response = Connections(snowflake_connection).create_connections()


        if not s3_conn_response and snowflake_conn_response:
            print("Connection not established!!")

   
    s3_hook = S3Hook(aws_conn_id = config["aws"]["connection_id"])


    scrap_raw_tweets_from_web_ = PythonOperator(
                                    task_id = "scrap_raw_tweets_from_web",
                                    python_callable = scrap_raw_tweets_from_web,
                                    op_kwargs = {
                                        's3_hook': s3_hook,
                                        'bucket_name': config["aws"]["s3_bucket_name"],
                                        'search_query': config["tweets-scraping"]["search_query"],
                                        'tweet_limit': config["tweets-scraping"]["tweet_limit"],
                                        'raw_file_name': config["files"]["raw_file_name"]
                                        }
                                    )

    @task(task_id = "download_from_s3")
    def download_data_from_s3_bucket(temp_data_path: str, file_name: str) -> None:
       
        downloaded_file = s3_hook.download_file(
                                            key = file_name,
                                            bucket_name = config["aws"]["s3_bucket_name"],
                                            local_path = temp_data_path
                                            )
        os.rename(src = downloaded_file, destination = f"{temp_data_path}/{file_name}")

    with TaskGroup(group_id = "sentiment_labelling") as group1:
      
        add_sentiment_labels_to_scrapped_tweets_ = PythonOperator(
                                                task_id = "add_sentiment_labels_to_scrapped_tweets",
                                                python_callable = add_sentiment_labels_to_tweets,
                                                op_kwargs = {
                                                    's3_hook': s3_hook,
                                                    'bucket_name': config["aws"]["s3_bucket_name"],
                                                    'temp_data_path': config["aws"]["temp_data_path"],
                                                    'raw_file_name': config["files"]["raw_file_name"],
                                                    'labelled_file_name': config["files"]["labelled_file_name"],
                                                }
                                            )

      
        download_data_from_s3_bucket(config["aws"]["temp_data_path"], config["files"]["raw_file_name"]) >> add_sentiment_labels_to_scrapped_tweets_


    with TaskGroup(group_id = "preprocess_tweets_using_NLP") as group2:
    
        preprocess_tweets_ = PythonOperator(
                                task_id = "preprocess_labelled_tweets_using_nlp_techniques",
                                python_callable = preprocess_tweets,
                                op_kwargs = {
                                    's3_hook': s3_hook,
                                    'bucket_name': config["aws"]["s3_bucket_name"],
                                    'temp_data_path': config["aws"]["temp_data_path"],
                                    'labelled_file_name': config["files"]["labelled_file_name"],
                                    'preprocessed_file_name': config["files"]["preprocessed_file_name"]
                                }
                            )
        
      
        download_data_from_s3_bucket(config["aws"]["temp_data_path"], config["files"]["labelled_file_name"]) >> preprocess_tweets_

    @task(task_id = "load_processed_data_to_datawarehouse")
    def load_processed_data_to_snowflake(processed_file: str, table_name: str) -> None:
      
        try:
           
            snowflake_conn = SnowflakeHook(
                                        snowflake_conn_id = "snowflake_conn",
                                        account = os.environ["ACCOUNT"],
                                        warehouse = os.environ["WAREHOUSE"],
                                        database = os.environ["DATABASE"],
                                        schema = os.environ["SCHEMA"],
                                        role = os.environ["ROLE"]
                                        )

            dataframe = load_dataframe(processed_file)

          
            write_pandas(
                        conn = snowflake_conn,
                        df = dataframe,
                        table_name = table_name,
                        quote_identifiers = False
                        )
        
        except Exception as exc:
            raise ConnectionError("Something went wrong with the snowflake connection. Please check them!!") from exc

        finally:
            snowflake_conn.close()

 
    set_connections() >> scrap_raw_tweets_from_web_>> group1 >> group2 >> load_processed_data_to_snowflake(config["files"]["preprocessed_file_name"], config["misc"]["table_name"])


etl_dag = twitter_data_pipeline_dag_etl()