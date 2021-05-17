from __future__ import division, print_function
import cv2
from tensorflow import keras
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# coding=utf-8
import sys
import os
import glob
import re

model = keras.models.load_model('digits_model.h5')

app = Flask(__name__)  
 
@app.route('/')  
def index():  
    return render_template("index.html")  
 

def prediction(filepath):
    image = cv2.imread(filepath)
    image = cv2.resize(image,(28,28))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image/255
    image = image.reshape(-1,28,28,1)
    predictions = model.predict(image)
    predict = np.argmax(predictions)
    return predict

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'images', secure_filename(f.filename))
        f.save(file_path)
        pred = prediction(file_path)
        result = str(pred)
        return result
    return None


if __name__ == '__main__':  
    app.run(debug = True,)  
