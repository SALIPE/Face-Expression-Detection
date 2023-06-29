FROM python:3.9-slim-buster

# set working directory
WORKDIR /usr/src/fer-back

RUN apt-get update && apt-get -y install libgl1 libglib2.0-0 libsm6 libxrender1 libxext6
# add and install requirements
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY ./apurated_model_alex.pth .
# add app
COPY . .

CMD ["python", "app.py"]