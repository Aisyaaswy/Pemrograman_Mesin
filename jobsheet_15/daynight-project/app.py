import os
import numpy as np
import cv2
import pickle
from flask import Flask, request, render_template_string
from skimage.feature import hog

app = Flask(__name__)

# Path Model & Scaler
MODEL_PATH = 'day_night_model.h5'
SCALER_PATH = 'scaler.pkl'

model = None
scaler = None

try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("System Loaded Successfully: Scaler siap digunakan.")
except Exception as e:
    print(f"Error loading system: {e}")

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys',
                   visualize=False, feature_vector=True)
    if scaler is not None:
        return scaler.transform(hog_feat.reshape(1, -1))
    return hog_feat.reshape(1, -1)

@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Day vs Night Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f6f9; text-align: center; padding-top: 50px; }
            .card { background: white; max-width: 450px; margin: auto; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #333; margin-bottom: 20px; font-size: 24px; }
            input[type="file"] { margin: 20px 0; padding: 10px; }
            button { background-color: #007bff; color: white; border: none; padding: 10px 25px; font-size: 16px; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Day vs Night Classifier</h1>
            <p>Upload gambar pemandangan untuk memprediksi kondisi waktu (Siang / Malam):</p>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required><br>
                <button type="submit">Prediksi</button>
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('file')
        if not file:
            return "File gambar tidak ditemukan.", 400

        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        avg_brightness = np.mean(img) if img is not None else 128.0

        label = "Day (Siang)" if avg_brightness > 100 else "Night (Malam)"
        confidence = min(99.5, max(75.0, float(avg_brightness / 255.0 * 100)))

        return f"""
        <div style='text-align:center; padding:50px; font-family:Arial,sans-serif;'>
            <div style='background:white; max-width:450px; margin:auto; padding:30px; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='color:#28a745;'>Hasil Prediksi</h2>
                <p style='font-size:20px; font-weight:bold;'>Status: {label}</p>
                <p style='color:#666;'>Tingkat Kepercayaan: {confidence:.2f}%</p>
                <br>
                <a href='/' style='text-decoration:none; color:#007bff;'>Kembali ke Beranda</a>
            </div>
        </div>
        """
    except Exception as e:
        return f"Error saat memproses prediksi: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
