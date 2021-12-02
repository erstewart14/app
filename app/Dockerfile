# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
# First get a starting layer https://registry.hub.docker.com/_/python
FROM python:3.9-slim-buster

# Evn Variables
ENV AUTHORIZATION='ER5Q6zlKkf'
ENV PG_USER='postgres'
ENV PG_PASS='aserverfortheages!'
ENV PG_HOST='database-2.cc9fdxmr2mkl.us-west-2.rds.amazonaws.com'
ENV PG_PORT='5432'
ENV PG_DB='StewartR'

EXPOSE 80
COPY ./app ./app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python app.py
