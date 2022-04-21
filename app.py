from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
import predict

app = Flask(__name__)
CORS(app, resources=r"/*")
@app.route('/')
def index():
    return "response ok!"

@app.route('/predict', methods=['GET','POST'])
def predict_images():

    data = request.files.get("file")
    if data == None:
        return 'No Files Attached'
    else:
        prediction = {"result" : predict.predict(data)}

    return json.dumps(prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)