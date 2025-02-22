# Submission 1: Predict Student Performance
Nama: Akhmad Ardiansyah Amnur

Username dicoding: akhmad_ardi

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Predict Student Performance](https://www.kaggle.com/datasets/stealthtechnologies/predict-student-performance-dataset) |
| Masalah | Salah satu tantangan terbesar dalam dunia pendidikan adalah mengidentifikasi siswa yang berisiko mendapatkan nilai rendah sebelum ujian berlangsung. Dengan adanya prediksi yang akurat mengenai performa siswa, guru dan institusi pendidikan dapat mengambil langkah-langkah preventif, seperti memberikan bimbingan tambahan atau metode pembelajaran yang lebih sesuai. Tanpa prediksi yang tepat, banyak siswa mungkin tidak mendapatkan intervensi yang dibutuhkan tepat waktu, sehingga berpotensi mengalami kesulitan akademik yang lebih besar di masa depan. Oleh karena itu, pengembangan model prediksi ini bertujuan untuk membantu pihak sekolah dalam memberikan perhatian yang lebih personal kepada siswa yang membutuhkan. |
| Solusi machine learning | Untuk membangun sistem prediksi performa akademik siswa, digunakan pendekatan machine learning berbasis TensorFlow yang dikombinasikan dengan TensorFlow Extended (TFX) untuk membangun pipeline yang terstruktur dan dapat di-deploy ke lingkungan produksi. |
| Metode pengolahan | Metode pengolahan data merupakan tahapan penting dalam memastikan kualitas data sebelum digunakan dalam pelatihan model machine learning. Pengolahan data dilakukan dengan memanfaatkan TensorFlow Extended (TFX), yang memungkinkan pipeline berjalan secara otomatis dan dapat diperbarui dengan data terbaru. |
| Arsitektur model | Model prediksi performa akademik siswa dibangun menggunakan arsitektur jaringan saraf tiruan (Artificial Neural Network - ANN) berbasis TensorFlow Keras. Model ini menerima beberapa fitur numerik yang telah dinormalisasi sebagai input, kemudian diproses melalui beberapa lapisan tersembunyi untuk menghasilkan output berupa prediksi nilai siswa. |
| Metrik evaluasi | Model ini dievaluasi menggunakan metrik Mean Absolute Error (MAE), Mean Squared Error (MSE), dan Root Mean Squared Error (RMSE). |
| Performa model | Model yang dikembangkan menunjukkan kinerja yang cukup baik dalam memprediksi nilai berdasarkan data kehadiran, durasi tidur, skor sosial ekonomi, dan jam belajar yang diberikan. Hasil pelatihan menunjukkan bahwa model memiliki Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE) yang rendah, yang mengindikasikan bahwa prediksi yang dihasilkan cukup akurat. |