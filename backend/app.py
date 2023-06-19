from flask import Flask,request,Response
import jsonpickle
from face_detector import get_facial_expression
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return Response(response=jsonpickle.encode({"emotion":"broxa"}), status=200, mimetype="application/json")

@app.post("/get-image")
def get_face():
    file = request.files['imagefile'].read()
    face_expression = get_facial_expression(file)
    response = {'message': face_expression}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == "__main__":
    app.run(host="0.0.0.0")