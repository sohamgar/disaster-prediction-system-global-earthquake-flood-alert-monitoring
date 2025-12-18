#!/usr/bin/env python
from flask import Flask, render_template, request
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from twilio.rest import Client
import os

app = Flask(__name__)

# Load the trained model once at startup
MODEL_PATH = "earthquake_model.h5"  # Your trained Keras model
model = load_model(MODEL_PATH)

# Twilio environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
TWILIO_TO = os.getenv("TWILIO_TO")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/earthquake')
def earthquake():
    return render_template('earthq.html')

@app.route('/predict', methods=['POST'])
def predict():
    def mapdateTotime(x):
        epoch = datetime(1970, 1, 1)
        try:
            dt = datetime.strptime(x, "%m/%d/%Y")
        except ValueError:
            dt = datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ")
        diff = dt - epoch
        return diff.total_seconds()

    # Read input from form
    lat = float(request.form['lat'])
    long = float(request.form['long'])
    depth = float(request.form['depth'])
    date = request.form['date']

    # Convert date to seconds
    date_sec = mapdateTotime(date)

    # Prepare input for prediction
    input_data = np.array([[lat, long, depth, date_sec]], dtype=np.float32)

    # Predict magnitude
    prediction = model.predict(input_data)
    magnitude = float(prediction[0][0])

    # Send Twilio alert if magnitude > 6
    if magnitude > 6 and all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, TWILIO_TO]):
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            to=TWILIO_TO,
            from_=TWILIO_FROM,
            body=f"⚠️ Earthquake Alert! Predicted magnitude: {magnitude:.2f}"
        )

    return render_template('earth.html', prediction=magnitude)


if __name__ == "__main__":
    app.run(debug=True)
