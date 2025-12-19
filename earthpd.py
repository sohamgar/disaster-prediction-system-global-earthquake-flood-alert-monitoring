#!/usr/bin/env python
from flask import Flask, render_template, request
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from twilio.rest import Client
import os

# --------------------------------
# Flask App
# --------------------------------
app = Flask(__name__)

# --------------------------------
# Load ML Model (IMPORTANT FIX)
# --------------------------------
MODEL_PATH = "earthquake_model.h5"

# compile=False fixes: keras.metrics.mse error
model = load_model(MODEL_PATH, compile=False)

# --------------------------------
# Twilio Config (ENV VARIABLES)
# --------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
TWILIO_TO = os.getenv("TWILIO_TO")

# --------------------------------
# Routes
# --------------------------------
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template("index.html", username=session.get('user_name'))

@app.route('/earthquake')
def earthquake():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template("earthq.html", username=session.get('user_name'))

@app.route('/flood')
def flood():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template("floods.html", username=session.get('user_name'))

@app.route('/login')
def login_page():
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))



@app.route("/predict", methods=["POST"])
def predict():
    # -----------------------------
    # Session protection
    # -----------------------------
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    # -----------------------------
    # Convert date → epoch seconds
    # -----------------------------
    def mapdateTotime(date_str):
        epoch = datetime(1970, 1, 1)
        try:
            dt = datetime.strptime(date_str, "%m/%d/%Y")
        except ValueError:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        return (dt - epoch).total_seconds()

    # -----------------------------
    # Get User Input
    # -----------------------------
    lat = float(request.form["lat"])
    lon = float(request.form["long"])
    depth = float(request.form["depth"])
    date = request.form["date"]

    # Normalize date lightly (model-safe)
    date_sec = mapdateTotime(date) / 1e9

    # Input shape must match training
    input_data = np.array([[lat, lon, depth, date_sec]], dtype=np.float32)

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(input_data, verbose=0)
    magnitude = float(prediction[0][0])

    # -----------------------------
    # Twilio Alert
    # -----------------------------
    if (
        magnitude > 6
        and TWILIO_ACCOUNT_SID
        and TWILIO_AUTH_TOKEN
        and TWILIO_FROM
        and TWILIO_TO
    ):
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            to=TWILIO_TO,
            from_=TWILIO_FROM,
            body=f"⚠️ Earthquake Alert! Predicted magnitude: {magnitude:.2f}",
        )

    return render_template("earth.html", prediction=magnitude)


# --------------------------------
# Run (LOCAL ONLY)
# --------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
