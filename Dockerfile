# start from python base image
FROM python:3.9

# change working directory
WORKDIR /code

# add requirements file to image
COPY ./requirements-app.txt /code/requirements.txt

# install python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# add python code
COPY . /code/

# specify default commands
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]