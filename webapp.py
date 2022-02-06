# additional program creates a flask web-app that reads data from a user form and prints the model prediction result

import numpy as np
import xgboost as xgb
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('index.html')


def get_form_input():
    input_names = ["age", "gender", "height_cm", "weight_kg", "body_fat_percentage", "diastolic", "systolic",
                   "gripForce", "sit_and_bend_forward_cm", "sit_ups_count", "broad_jump_cm"]
    input_values = []
    for input_name in input_names:
        input_values.append(float(request.form[input_name]))
    return input_values


@app.route('/', methods=['POST'])
def my_form_post():
    model = xgb.XGBClassifier()
    model.load_model("model.json")
    # print(get_form_input())
    form_input = get_form_input()
    print(form_input)
    form_input = np.array(form_input).reshape((1, -1))
    prediction = model.predict(form_input)
    prediction = list(prediction)[0]
    if prediction == 0:
        prediction = "A"
    elif prediction == 1:
        prediction = "B"
    elif prediction == 2:
        prediction = "C"
    elif prediction == 3:
        prediction = "D"
    return "<h2 align='center' size='16'>Class of the user is " + prediction + "</h2>"

# (Windows) To run flask server, cd to project's dir, run the following commands in Terminal:
# $env:FLASK_APP = "webapp"
# python -m flask run
