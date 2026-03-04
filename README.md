# MBatikin - Sistem Klasifikasi Motif Batik Indonesia
MBatikin adalah sistem berbasis kecerdasan buatan yang digunakan untuk mengklasifikasikan motif batik Indonesia secara otomatis menggunakan teknologi **Deep Learning**. Sistem ini memanfaatkan metode **Convolutional Neural Network (CNN)** dengan arsitektur **DenseNet121** untuk mengenali pola visual dari citra batik dan mengelompokkannya ke dalam kategori motif tertentu.
Proyek ini dikembangkan untuk membantu proses identifikasi motif batik yang biasanya memerlukan pengetahuan khusus tentang batik. Dengan sistem ini, pengguna dapat mengunggah gambar batik melalui website dan mendapatkan hasil klasifikasi motif secara cepat dan otomatis.

## Fitur Utama
* Klasifikasi otomatis motif batik berbasis citra
* Menggunakan model **CNN DenseNet121 dengan transfer learning**
* Mendukung **5 kelas motif batik**:
  * Parang
  * Kawung
  * Mega Mendung
  * Batik Bali
  * Batik Betawi
* Website berbasis **Flask** untuk melakukan prediksi secara langsung
* Menampilkan **probabilitas dan confidence score** hasil klasifikasi
* Evaluasi model menggunakan **Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix**

## Teknologi yang Digunakan
* Python
* TensorFlow & Keras
* DenseNet121 (Transfer Learning)
* Flask
* HTML, CSS, JavaScript
* Google Colaboratory

## Cara Kerja Sistem
1. Pengguna mengunggah gambar motif batik melalui website.
2. Sistem melakukan preprocessing pada gambar (resize, normalisasi).
3. Model CNN melakukan ekstraksi fitur dari citra batik.
4. Model memprediksi motif batik menggunakan fungsi **Softmax**.
5. Sistem menampilkan hasil klasifikasi beserta tingkat probabilitas setiap kelas.

## Dataset
Dataset yang digunakan berasal dari dataset publik **Indonesian Batik Motifs** yang berisi berbagai citra motif batik Indonesia.

## Tujuan Proyek
Proyek ini bertujuan untuk:
* Mengembangkan sistem klasifikasi motif batik berbasis deep learning.
* Membantu masyarakat mengenali motif batik secara otomatis.
* Mendukung digitalisasi dan pelestarian budaya batik Indonesia melalui teknologi kecerdasan buatan.

## Hasil Model
Model CNN dengan arsitektur DenseNet121 yang dilatih pada dataset batik mampu mencapai **akurasi sekitar 88%** dalam mengklasifikasikan motif batik Indonesia. 


