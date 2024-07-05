import os
import sys
import mlflow
from mlflow import sagemaker

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import Config

config = Config()

mlflow.set_tracking_uri(config["model-tracking"]["mlflow_tracking_uri"])


app_name = config["model-deploy"]["endpoint_name"]


model_name = config["model-registry"]["model_name"]
model_uri = f"models:/{model_name}/production"

docker_image_url = os.environ["IMAGE_URI"]


role = os.environ["ARN_ROLE"]

region = os.environ["REGION"]

sagemaker._deploy(
                mode = 'create',
                app_name = app_name,
                model_uri = model_uri,
                image_url = docker_image_url,
                execution_role_arn = role,
                instance_type = 'ml.m5.xlarge',
                instance_count = 1,
                region_name = region
            )