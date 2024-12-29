
from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd


import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from preprocessing_utility import normalize_text

# Initializing Flask.
app = Flask(__name__)

# We need to load the model outside, because we don't want to call our model multiples time from model registry. suppose, user click 1 crores time in a day, that does not mean, we want to load our model 1 crores time so. that's why we need to load the model outside the predict function.
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/netra200021kcbdr/mlops-mini-project.mlflow")
dagshub.init(repo_owner='netra200021kcbdr', repo_name='mlops-mini-project', mlflow=True)

# load model from model registry.
model_name = 'my_model'
model_version = 1

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html", result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Clean text from the user entered. 
    text = normalize_text(text)

    # BoW. 
    features = vectorizer.transform([text])

    # Show. 
    result = model.predict(features)

    return render_template("index.html", result=result[0])

app.run(debug=True)