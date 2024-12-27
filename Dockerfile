FROM python:3.11-buster
ENV APPLICATION_SERVICE=/app

# set work directory
RUN mkdir -p $APPLICATION_SERVICE

# where the code lives
WORKDIR $APPLICATION_SERVICE

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
COPY poetry.lock pyproject.toml ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev
# copy project
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY . $APPLICATION_SERVICE


CMD streamlit run ai_test/app.py