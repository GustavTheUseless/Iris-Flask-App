import numpy as np
from flask import Flask, request, render_template, redirect
import pickle
# define Flask app
app = Flask(__name__)
# load machine learning model
model = pickle.load(open("../iris-model.pkl", "rb"))
# define initial route and returns the index.html page 
# aswell as sets the parameter 'flower_name' equal to an empty string
@app.route("/")
def home():
    return render_template("index.html", flower_name="")

# define route 'classify' which only accept POST requests
# I define an array named values which contains all 
# the values from the from in index.html
# I define a variable named prediction which contains the
# prediction from the machine learning model
@app.route("/classify", methods=["POST"])
def predict():
    values = [float(x) for x in request.form.values()]
    prediction = model.predict([values])
    return render_template("index.html", flower_name=prediction[0])

# runs the app at port 6969
if __name__ == "__main__":
    app.run(port=6969)
