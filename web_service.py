from flask import Flask,request,Response
import jsonpickle
from face_detector import face_img_detector
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.post("/get-image")
def get_face():
    file = request.files['imagefile'].read()
    face_img_detector(file)
    response = {'message': 'image received. filename={}'.format(file.filename)}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")