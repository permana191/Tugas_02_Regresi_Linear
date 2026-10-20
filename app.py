import os
import json
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load Model & Metrik
MODEL_PATH = os.path.join('models', 'model_likes.joblib')
METRICS_PATH = 'metrics.json'

model = None
metrics_info = {
    "intercept": 25.12,
    "slope": 0.32,
    "mae": 112.5,
    "mse": 24500.0,
    "rmse": 156.5,
    "r2": 0.107,
    "corr": 0.327,
    "dataset_count": 99003,
    "mean_likes_given": 156.4,
    "mean_likes_received": 131.2
}

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, 'r') as f:
        metrics_info = json.load(f)

def classify_behavior(likes_given, prediction, ratio):
    """Pengklasifikasi profil psikologi interaksi sosial"""
    if likes_given == 0:
        return {
            "title": "Pasif Observer",
            "badge": "bg-secondary",
            "desc": "Anda cenderung menikmati konten tanpa berinteraksi aktif. Prediksi likes diterima murni didorong oleh faktor eksternal dasar."
        }
    elif ratio > 100:
        return {
            "title": "High ROI Socialite",
            "badge": "bg-success",
            "desc": "Konten Anda sangat menarik! Anda mendapatkan pengembalian likes jauh lebih tinggi dibanding likes yang Anda berikan."
        }
    elif ratio >= 40:
        return {
            "title": "Balanced Reciprocator",
            "badge": "bg-primary",
            "desc": "Interaksi sosial Anda sehat dan seimbang. Terdapat timbal balik sosial yang stabil dalam jejaring pertemanan Anda."
        }
    else:
        return {
            "title": "Generous Supporter",
            "badge": "bg-warning text-dark",
            "desc": "Anda sangat murah hati memberikan dukungan likes. Meskipun rasio timbal balik lebih rendah, peran Anda penting dalam menghidupkan jejaring."
        }

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', metrics=metrics_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        likes_given = float(request.form.get('likes_given', 0))
        likes_given = max(0, likes_given)

        # Proses Prediksi
        if model is not None:
            pred_raw = model.predict(np.array([[likes_given]]))[0]
        else:
            # Fallback jika file model belum di-train
            pred_raw = metrics_info['intercept'] + (metrics_info['slope'] * likes_given)

        prediction = max(0, round(float(pred_raw)))

        # Hitung Rasio Timbal Balik
        ratio = round((prediction / likes_given * 100), 2) if likes_given > 0 else 0.0

        # Margin Error Prediksi (± RMSE)
        lower_bound = max(0, round(prediction - metrics_info['rmse']))
        upper_bound = round(prediction + metrics_info['rmse'])

        # Klasifikasi Perilaku
        behavior = classify_behavior(likes_given, prediction, ratio)

        return render_template(
            'result.html',
            likes_given=int(likes_given),
            prediction=prediction,
            ratio=ratio,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            behavior=behavior,
            metrics=metrics_info
        )

    except Exception as e:
        return f"Terjadi kesalahan saat memproses data: {str(e)}", 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API Endpoint untuk kebutuhan panggilan AJAX/Live Preview"""
    data = request.get_json()
    try:
        likes_given = float(data.get('likes_given', 0))
        if model is not None:
            pred_val = model.predict(np.array([[likes_given]]))[0]
        else:
            pred_val = metrics_info['intercept'] + (metrics_info['slope'] * likes_given)
        
        prediction = max(0, round(float(pred_val)))
        ratio = round((prediction / likes_given * 100), 2) if likes_given > 0 else 0.0

        return jsonify({
            "status": "success",
            "likes_given": likes_given,
            "predicted_likes_received": prediction,
            "reciprocity_ratio_percent": ratio
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)