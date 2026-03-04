import os
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input

app = Flask(__name__)

# --- KONFIGURASI ---
# Ganti nama folder ini sesuai dengan folder hasil ekstrak ZIP Anda
MODEL_PATH = 'model_batik_raw' 

print(f"Sedang memuat model RAW dari folder: {MODEL_PATH}...")
try:
    # Memuat model format SavedModel
    loaded_model = tf.saved_model.load(MODEL_PATH)
    inference_func = loaded_model.signatures['serving_default']
    print("✅ Model RAW BERHASIL dimuat!")
except Exception as e:
    print(f"❌ Error fatal saat load model: {e}")
    print("Tips: Pastikan folder hasil ekstrak ada di sebelah file app.py")
    inference_func = None

# Urutan kelas (Harus sama persis dengan urutan di Colab)
class_names = ['batik-bali', 'batik-betawi', 'batik-kawung', 'batik-megamendung', 'batik-parang']

def predict_serving_densenet(img_path, infer_func):
    # 1. Preprocessing Gambar
    # PENTING: interpolation='nearest' agar cara resize sama persis dengan Colab
    img = image.load_img(img_path, target_size=(224, 224), interpolation='nearest')
    
    # Ubah ke array
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    # Preprocess Input ala DenseNet
    x = preprocess_input(x)
    
    # 2. Prediksi (Inference)
    # KARENA MODEL INI ADALAH MODEL RAW, OUTPUTNYA ADALAH LOGITS (NILAI Z)
    input_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    prediction_dict = infer_func(input_tensor)
    
    # Ambil output tensor (biasanya key pertama)
    output_tensor = list(prediction_dict.values())[0]
    
    # Ambil Logits Mentah (Contoh: 2.99, -1.93, dll)
    logits = output_tensor.numpy()[0]
    
    # 3. Hitung Softmax Manual (Agar dapat Persentase %)
    # Rumus: e^z / sum(e^z)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits)
    
    results = []
    for i, raw_logit in enumerate(logits):
        if i < len(class_names):
            label = class_names[i]
            
            # A. AMBIL NILAI LOGIT ASLI (Untuk Tabel Bawah)
            # Ini angka yang Anda cari-cari (sama dengan Colab)
            z_value = float(raw_logit)
            
            # B. AMBIL PERSENTASE (Untuk Grafik Bar)
            prob_percent = round(probabilities[i] * 100, 2)
            
            results.append({
                'label': label, 
                'score': prob_percent,     # Masuk ke Bar Chart
                'logit': round(z_value, 4) # Masuk ke Tabel Logits
            })
    
    # Urutkan dari confidence tertinggi
    results.sort(key=lambda x: x['score'], reverse=True)
    
    top_pred = results[0]['label']
    top_conf = results[0]['score']
    
    return top_pred, top_conf, results

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    all_results = None
    img_url = None
    error_msg = None

    if request.method == 'POST':
        if not inference_func:
            return render_template('index.html', error="Model Belum Siap / Gagal Dimuat")
            
        file = request.files['file']
        if file.filename != '':
            if not os.path.exists('static'):
                os.makedirs('static')
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            
            try:
                # Panggil fungsi prediksi
                pred, conf, results_list = predict_serving_densenet(file_path, inference_func)
                
                prediction = pred
                confidence = conf
                all_results = results_list
                img_url = file_path
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                error_msg = "Gagal memproses gambar."

    return render_template('index.html', 
                           prediction=prediction, 
                           confidence=confidence, 
                           all_results=all_results, 
                           img_url=img_url, 
                           error=error_msg)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)