FROM python:3.9-slim-buster

# set working directory
WORKDIR /usr/src/fer

RUN apt-get update && apt-get -y install libgl1 libglib2.0-0 libsm6 libxrender1 libxext6
# add and install requirements
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# add app
COPY . .

CMD ["python", "app.py"]