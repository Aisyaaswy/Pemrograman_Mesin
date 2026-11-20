import os
import joblib
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Path Model & Scaler
MODEL_PATH = 'sleep_health_model.pkl'
SCALER_PATH = 'sleep_health_scaler.pkl'

model = None
scaler = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("System Loaded Successfully: Model dan Scaler siap digunakan.")
except Exception as e:
    print(f"Error loading system: {e}")

@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sleep Health & Lifestyle Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f0f2f5; padding: 30px; margin: 0; }
            .container { max-width: 520px; background: white; margin: auto; padding: 30px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
            h2 { color: #1a365d; text-align: center; margin-bottom: 8px; }
            p.desc { text-align: center; color: #666; font-size: 14px; margin-bottom: 25px; }
            .form-group { margin-bottom: 15px; }
            label { display: block; font-weight: bold; margin-bottom: 5px; color: #333; font-size: 13px; }
            input[type="number"], input[type="text"] { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 6px; box-sizing: border-box; }
            button { width: 100%; background: #2b6cb0; color: white; padding: 12px; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; font-weight: bold; margin-top: 10px; }
            button:hover { background: #2c5282; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Sleep Health Classifier</h2>
            <p class="desc">Deteksi Dini Gangguan Tidur (Normal vs Gangguan Tidur)</p>
            <form action="/predict" method="post">
                <div class="form-group">
                    <label>Usia (Tahun):</label>
                    <input type="number" name="Age" value="30" required>
                </div>
                <div class="form-group">
                    <label>Durasi Tidur (Jam/Hari):</label>
                    <input type="number" step="0.1" name="Sleep_Duration" value="7.0" required>
                </div>
                <div class="form-group">
                    <label>Kualitas Tidur (Skala 1 - 10):</label>
                    <input type="number" name="Quality_of_Sleep" min="1" max="10" value="7" required>
                </div>
                <div class="form-group">
                    <label>Tingkat Aktivitas Fisik (Menit/Hari):</label>
                    <input type="number" name="Physical_Activity_Level" value="60" required>
                </div>
                <div class="form-group">
                    <label>Tingkat Stres (Skala 1 - 10):</label>
                    <input type="number" name="Stress_Level" min="1" max="10" value="5" required>
                </div>
                <div class="form-group">
                    <label>Denyut Jantung (BPM):</label>
                    <input type="number" name="Heart_Rate" value="70" required>
                </div>
                <div class="form-group">
                    <label>Langkah Harian (Daily Steps):</label>
                    <input type="number" name="Daily_Steps" value="7000" required>
                </div>
                <button type="submit">Prediksi Kondisi Tidur</button>
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form.get('Age')),
            float(request.form.get('Sleep_Duration')),
            float(request.form.get('Quality_of_Sleep')),
            float(request.form.get('Physical_Activity_Level')),
            float(request.form.get('Stress_Level')),
            float(request.form.get('Heart_Rate')),
            float(request.form.get('Daily_Steps'))
        ]

        features_arr = np.array(data).reshape(1, -1)
        features_scaled = scaler.transform(features_arr)

        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]

        if pred == 0:
            label = "Normal (Pola Tidur Sehat)"
            confidence = proba[0] * 100
            color = "#28a745"
        else:
            label = "Gangguan Tidur (Terindikasi Insomnia / Sleep Apnea)"
            confidence = proba[1] * 100
            color = "#dc3545"

        return f"""
        <div style='text-align:center; padding:50px; font-family:Arial,sans-serif; background:#f0f2f5; min-height:100vh;'>
            <div style='background:white; max-width:480px; margin:auto; padding:35px; border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.1);'>
                <h2 style='color:#1a365d; margin-bottom:15px;'>Hasil Evaluasi Kesehatan Tidur</h2>
                <div style='padding:15px; border-radius:8px; background:#f8f9fa; border-left: 5px solid {color}; margin-bottom:20px;'>
                    <p style='font-size:18px; font-weight:bold; color:{color}; margin:0 0 8px 0;'>Status: {label}</p>
                    <p style='color:#555; margin:0;'>Tingkat Kepercayaan: <b>{confidence:.2f}%</b></p>
                </div>
                <a href='/' style='display:inline-block; text-decoration:none; background:#2b6cb0; color:white; padding:10px 20px; border-radius:6px; font-weight:bold;'>Kembali ke Form</a>
            </div>
        </div>
        """
    except Exception as e:
        return f"Error saat memproses prediksi: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
