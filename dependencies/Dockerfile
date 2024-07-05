FROM apache/airflow:2.4.1-python3.9

COPY ./dependencies/requirements.txt /requirements.txt

RUN pip install --user --upgrade pip

RUN pip install -r /requirements.txt

RUN --mount=type=secret,id=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=AWS_SECRET_ACCESS_KEY \
    --mount=type=secret,id=REGION \
    --mount=type=secret,id=LOGIN \
    --mount=type=secret,id=PASSWORD \
    --mount=type=secret,id=HOST \
    --mount=type=secret,id=ACCOUNT \
    --mount=type=secret,id=WAREHOUSE \
    --mount=type=secret,id=DATABASE \
    --mount=type=secret,id=SCHEMA \
    export AWS_ACCESS_KEY_ID=$(cat /run/secrets/AWS_ACCESS_KEY_ID) && \
    export AWS_SECRET_ACCESS_KEY=$(cat /run/secrets/AWS_SECRET_ACCESS_KEY) && \
    export REGION=$(cat /run/secrets/REGION) && \
    export LOGIN=$(cat /run/secrets/LOGIN) && \
    export PASSWORD=$(cat /run/secrets/PASSWORD) && \
    export HOST=$(cat /run/secrets/HOST) && \
    export ACCOUNT=$(cat /run/secrets/ACCOUNT) && \
    export WAREHOUSE=$(cat /run/secrets/WAREHOUSE) && \
    export DATABASE=$(cat /run/secrets/DATABASE) && \
    export SCHEMA=$(cat /run/secrets/SCHEMA)