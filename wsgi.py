import sys
import os

path = "/home/soham-123/www/Disaster-Prediction-main"
if path not in sys.path:
    sys.path.insert(0, path)

from prediction import app  # ya app.py ka naam

application = app
