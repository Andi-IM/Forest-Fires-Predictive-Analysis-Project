# Laporan Proyek Machine Learning - Andi Irham M

## Domain Proyek

Kebakaran hutan/lahan merupakan salah satu bencana yang setiap tahunnya terjadi di beberapa negara di dunia. Peristiwa ini mendapat perhatian lebih dari pemerintah karena menimbulkan banyak kerugian baik pada bidang ekonomi, ekologi, dan sosial. Kebakaran hutan/lahan sering terjadi akibat penggunaan api dalam pembukaan hutan/lahan untuk difungsikan sebagai Hutan Tanaman Industri (HTI), perkebunan kelapa sawit, pertanian serta pembalakan liar[^1].  Dengan masifnya persebaran titik api yang tidak menentu membuat pemangku kebijakan akan kesulitan dalam menanggulangi dampak dari kebakaran hutan. Untuk itu, diperlukan metode yang cocok untuk memprediksi kemungkinan-kemungkinan terjadinya kebakaran hutan berdasarkan lokasi. Dataset yang digunakan untuk menganalisis berasal dari penelitian tahun 2008 oleh Paulo Cortez[^2] dan memprediksi hal-hal apa saja yang berpotensi terjadinya kebakaran hutan.  

## Business Understanding

Di sini kita perlu memahami tujuan apa yang akan kita raih melalui proyek ini, maka didapatkan beberapa problem, goal, dan solution statement sebagai berikut:

### Problem Statement

Berasarkan dari kondisi yang telah diuraikan sebelumnya, maka diperlukan pengembangan sistem yang dapat memprediksi kemungkinan terjadinya kebakaran hutan dengan menjawab permasalahan berikut:

- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh dalam memprediksi kemungkinan kebakaran hutan?
- Seberapa tinggi kemungkinan terjadinya kebakaran berdasarkan karakteristik atau fitur tertentu?

### Goal Statement

Untuk menjawab problem tersebut, maka akan dibuat predictive modeling dengan tujuan atau goals sebagai berikut:

- Mengetahui fitur yang paling penting dan berkolerasi dengan nilai kemungkinan terjadinya kebakaran hutan berdasarkan data citra.
- Membuat model machine learning yang dapat memprediksi kemungkinan terjadinya kebakaran berdasarkan fitur-fitur yang ada.  

### Solution Statement

Solusi yang dapat menjawab permasalahan dan tujuan adalah sebagai berikut:

- Melakukan eksplorasi fitur dengan menggunakan analisis univariat dan multivariat.
- Melakukan Data Wragling yang meliputi Data Gathering, Data assesing, dan Data Cleaning untuk mendapatkan model dengan performa yang baik.
- Menggunakan metode *Regresi* dengan memanfaatkan algoritma machine learning seperti KNN, Random Forest, Boosting, dan SVM.
- Mengunakan ***Root Mean Squared Error*** sebagai metrik untuk melihat akurasi dari model yang akan dibangun.  

## Data Understanding

Dataset yang digunakan dalam analisis kali ini adalah [*Fires from Space: Australia*](https://www.kaggle.com/datasets/carlosparadis/fires-from-space-australia-and-new-zeland) yang merupakan data kebakaran hutan yang diambil dari Satelit NASA MODISC6 dan VIIRS 375m pada tahun 2019 hingga 2020 yang terdokumentasi melalui platform [kaggle.com](kaggle.com).
Dataset ini berisi 3 tabel dari 2 instrumen NASA:

- Tabel MODIS C6:
  - fire_archive_M6_96619.csv
  - fire_nrt_M6_96619.csv
- Tabel VIIRS 375m:
  - fire_archive_V1_96617.csv
  - fire_nrt_96617.csv

Dalam analisis kali ini, saya menggunakan tabel yang berada pada file `fire_archive_M6_96619.csv`. Detail dari file ini adalah sebagai berikut:

- Dataset terdiri dari 36.011 *records* dengan 15 fitur.
- Dataset terdiri dari 3 data kategori dan 11 data numerik.
- Tidak ada data yang missing, namun struktur data kurang rapi.

**Dataset `fire_archive_M6_96619.csv` memiliki variabel-variabel sebagai berikut:**

- latitude: Posisi piskel dengan jarak 1 kilometer yang menandai lokasi aktual api pada peta.
- longitude: Posisi piksel dengan jarak 1 kilometer yang menandai lokasi aktual api pada peta.
- brightness: Temperatur kecerahan pada  
- scan: Ukuran piksel pemindaian citra.
- track: Ukuran piksel jalur pemindaian citra.
- acq_date: Tanggal penangkapan citra.
- acq_time: Waktu penangkapan citra.
- satellite: Satelit (A = Aqua, T = Terra)
- instrument: Instrumen yang digunakan, statis dengan nilai MODIS
- confidence: Tingkat Keyakinan (0 - 100%) berdassarkan algoritma pendeteksian yang digunakan.
- version: Versi yang mengidentifikasi koleksi
- bright_t31: Temperatur kecerahan piksel pada channel 31 diukur dengan satuan Kelvin.
- frp: Fire Radiative Power (MW - megawatt), kekuatan radiasi api dalam satuan megawatt
- daynight: Siang (D) atau Malam (N)
- type: tipe titik panas yang diinferensi, 0 = wilayah vegetasi, 1 = gunung aktif, 2 = lahan statis, 3 = lepas pantai

Lisensi dapat dilihat pada bagian referensi[^3][^4].

## Data Preparation

## Modeling

Model yang digunakan untuk memprediksi kemungkinan kebakaran berdasarkan data satelit antara lain:

- *K-Nearest Neighbors*
  
  K-Nearest Neighbours (kNN) adalah algoritma yang paling simple. Metode ini bekerja dengan cara mencari sejumlah *k* pola (di antara semua pola latih yang ada di semua kelas) yang terdekat dengan pola masukan, kemudian menentukan kelas keputusan berdasarkan jumlah pola terbanyak di antara *k* pola tersebut (voting). KNN dapat digunakan untuk kasus klasifikasi maupun regresi.

   **Tahapan Cara Kerja kNN**
  - Menentukan jumlah tetangga terdekat *k*.
  - Menghitung jarak antara data testing ke data training.
  - Mengurutkan data berdasarkan data yang mempunyai jarak terkecil (bisa menggunakan manhattan, eucledian ataupun minkowski)
  - Menentukan kelompok testing berdasarkan label pada *k*.

  Pada proyek ini menggunakan *n_neighbors = 50* dengan catatan pemilihan nilai *k* sangat penting dan berpengaruh dengan performa model. Metrik jarak juga memiliki keunggulan masing-masing, dan di proyek ini akan menggunakan metode Eucledian untuk menghitung jarak. Dan metode evaluasi selanjutnya akan dibahas pada [Evaluasi Model](#evaluasi-model) .

  **Kelebihan kNN**
  - *Mudah digunakan* dengan kompleksitas algoritma yang tidak sebegitunya tinggi.
  - *Mudah beradaptasi* - Algoritma ini menyimpan seluruh data dalam penyimpanan memori. Ketika sebuah contoh baru atau titik data ditambahkan, kNN secara otomatis menyesuaikan diri berdasarkan contoh baru dan turut berkontribusi pada prediksi masa depan.
  - *Sedikit pengaturan hyperparameter* -  Dalam training algoritma ini hanya memerlukan parameter k

  **Kekurangan kNN**
  - *Tidak dapat diskalakan* - Algoritma kNN sering juga dikatakan algoritma pemalas (*Lazy Algorithm*) karena ia tidak secara eksplisit mempelajari model dari data pelatihan. Akibatnya, kNN membutuhkan daya komputasi dan penyimpanan data selama fase prediksi. Hal ini membuat kNN memakan waktu dan sumber daya.
  - *Kutukan Dimensionalitas* - Algoritma kNN rentan terhadap "fenomena puncak" yang terkait dengan kutukan dimensionalitas. Ini berarti kNN menghadapi kesulitan dalam mengklasifikasikan titik data secara akurat ketika dimensi data menjadi terlalu tinggi.
  - *Rentan Overfit* - karena rentan dengan "Kutukan dimensionalitas", algoritma kNN juga rentan dengan masalah overfitting. Oleh karena itu, teknik pemilihan fitur dan reduksi dimensionalitas umumnya diterapkan untuk masalah ini.

- *Random Forest*
  
  Algoritma Random Forest adalah algoritma yang sering digunakan karena sederhana dan memiliki stabilitas yang mumpuni. Algoritma ini termasuk varian teknik *bagging*. Algoritma ini merupakan kombinasi pohon keputusan sedemikian hingga setiap pohon bergantung pada nilai vektor acak yang disampling secara independen dan dengan distribusi yang sama untuk semua pohon dalam hutan tersebut. Kekuatan random forest terletak pada seleksi fitur yang acak untuk memilah setiap *node*, yang mampu menghasilkan tingkat kesalahan relatif rendah..
- *Boosting*

  Boosting dikemukakan oleh Robert E. Schapire pada tahun 1990. Sesuai dengan namanya, metode boosting bekerja dengan cara memperkuat (*boost*) sebuah model klasifikasi awal yang lemah, secara sekuensial menggunakan penyamplingan objek data bootstrap berdasarkan pembobotan dinamis.
  Algoritma boosting sudah ada sejak puluhan tahun lalu. Kembali terkenal sejak adanya peningkatan dalam kompetisi machine learning atau data science. Algoritma ini sangat powerful dalam meningkatkan akurasi prediksi. Algoritma boosting sering mengungguli model yang lebih sederhana seperti logistic regression dan random forest. Beberapa pemenang kompetisi kaggle menyatakan bahwa mereka menggunakan algoritma boosting atau kombinasi beberapa algoritma boosting dalam modelnya. Meskipun demikian, hal ini tetap bergantung pada kasus per kasus, ruang lingkup masalah, dan dataset yang digunakan.
Dilihat caranya berkembang, algoritma boosting terdiri dari dua metode:
  - Adaptive boosting
  - Gradient boosting
- *SVM*

## Evaluation

Proyek ini menggunakan machine learning dengan kasus regresi oleh karena itu metrik yang digunakan adalah metrik yang membandingkan hasil prediksi dengan nilai sebenarnya. Model dikatakan baik jika memiliki nilai error yang kecil atau perbandingan antara hasil prediksi dengan nilai sebenarnya tidak jauh atau mendekati.

Root Mean Squared Error atau disingkat RMSE digunakan dengan menghitung nilai akar dari rata-rata kuadrat perbedaan antara nilai prediksi dengan nilai sebenarnya di dataset. RMSE didefenisikan sebagai persamaan berikut:

$$
\begin{align}
RMSE = \sqrt{\dfrac{1}{n}\Sigma(\hat y_i - y_i)^2}
\end{align}
$$

$Keterangan:$

- $N$ = jumlah dataset
- $\hat y_i$ = nilai prediksi
- $y_i$ = nilai sebenarnya

## Kesimpulan

Dapat dilihat dari keempat model yang digunakan dapat disimpulkan model random forest memiliki nilai error yang kecil.

## References

[^1]: D. F. Pramesti, M. T. Furqon, dan C. Dewi, “Implementasi Metode K-Medoids Clustering Untuk Pengelompokan Data Potensi Kebakaran Hutan/Lahan Berdasarkan Persebaran Titik Panas (Hotspot)”, J-PTIIK, vol. 1, no. 9, hlm. 723–732, Jun 2017.
[^2]: Cortez,Paulo and Morais,Anbal. (2008). Forest Fires. UCI Machine Learning Repository. <https://doi.org/10.24432/C5D88D>.
[^3]: NRT VIIRS 375 m Active Fire product VNP14IMGT. Available on-line [https://earthdata.nasa.gov/firms]. doi: <https://doi.org/10.5067/FIRMS/VIIRS/VNP14IMGT.NRT.001>.
[^4]: MODIS Collection 6 NRT Hotspot / Active Fire Detections MCD14DL. Available on-line [https://earthdata.nasa.gov/firms]. doi: <https://doi.org/10.5067/FIRMS/MODIS/MCD14DL.NRT.006>
