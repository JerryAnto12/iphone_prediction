import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, render_template,request
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle#Initialize the flask App
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict', methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    if prediction == 0:
        pred = "The customer is not going to buy iphone"
    elif prediction == 1:
        pred = "The customer may buy iphone11"
    elif prediction == 2:
        pred = "The customer may buy iphone12"
    elif prediction == 3:
        pred = "The customer may buy iphone13"
    elif prediction == 4:
        pred = "The customer may buy iphone13 pro"
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)