# Laporan Proyek Machine Learning - Saif Rayhan Naufal
#### _Rekomendasi Destinasi Wisata di Indonesia_

![Cover](https://github.com/user-attachments/assets/41f74ddd-8909-447e-9221-c7fcda7ba93a)

# Project Overview
Indonesia sebagai negara kepulauan terbesar di dunia memiliki berbagai destinasi wisata yang sangat beragam, mulai dari pantai eksotis, gunung berapi yang menakjubkan, hingga kekayaan budaya yang luar biasa. Sektor pariwisata berkontribusi signifikan terhadap perekonomian nasional dengan sektor ini menyumbang sekitar 3,8% dari Produk Domestik Bruto (PDB) Indonesia pada tahun 2023 (_[Kemenparekraf, 2024]_). Namun, meskipun memiliki potensi yang besar, industri pariwisata Indonesia masih menghadapi tantangan dalam menarik wisatawan, baik domestik maupun internasional untuk mengeksplorasi destinasi yang ada.

Salah satu tantangan utama adalah bagaimana wisatawan dapat menemukan destinasi yang sesuai dengan preferensi mereka. Banyak wisatawan kesulitan dalam memilih tempat wisata yang sesuai dengan minat pribadi mereka, seperti berdasarkan kategori yang disukai misalnya, wisata alam, budaya, dan sejarah atau berdasarkan faktor seperti wilayah dimana wisata tersebut berada. Selain itu, informasi yang tersebar di berbagai platform tidak selalu terintegrasi dengan baik, sehingga wisatawan sering kali merasa bingung dalam menentukan pilihan. Oleh karena itu, penerapan sistem rekomendasi berbasis teknologi bisa diadaptasi untuk mengatasi kebingungan wisatawan. Sistem rekomendasi yang memanfaatkan machine learning dapat membantu wisatawan menemukan destinasi yang sesuai dengan preferensi mereka secara lebih efektif dan efisien. 

# Business Understanding
## Problem Statement
1. Bagaimana memberikan rekomendasi destinasi wisata berdasarkan karakteristik yang serupa dengan destinasi yang pernah dikunjungi wisatawan?
2. Bagaimana sistem dapat memanfaatkan data rating untuk menghasilkan rekomendasi destinasi wisata yang relevan bagi wisatawan?

## Goals
1. Mengembangkan sistem rekomendasi yang dapat menganalisis karakteristik antar destinasi wisata untuk memberikan rekomendasi yang relevan dan sesuai dengan preferensi wisatawan.
2. Memanfaatkan data rating untuk menentukan destinasi wisata sehingga mampu memberikan rekomendasi yang lebih personal dan akurat.

## Solution Statements
1. Memanfaatkan algoritma berbasis content-based filtering untuk menemukan destinasi dengan karakteristik yang relevan. Sistem ini memanfaatkan atribut-atribut seperti kategori wisata dan lokasi wisata. Perbandingan dilakukan dengan menggunakan TF-IDF untuk merepresentasikan atribut dalam bentuk vektor numerik, sedangkan kesamaan antar destinasi dihitung menggunakan metode seperti cosine similarity.
2. Memanfaatkan collaborative filtering untuk merekomendasikan destinasi wisata berdasarkan rating yang diberikan oleh pengguna dengan preferensi serupa. Collaborative filtering bertujuan untuk menganalisis pola rating yang diberikan oleh wisatawan agar bisa menemukan hubungan preferensi antara pengguna. Metode ini bekerja dengan mengasumsikan bahwa wisatawan dengan kesukaan serupa cenderung memberikan penilaian yang sama pada destinasi tertentu. Salah satu metrik yang digunakan dalam proses ini adalah Root Mean Squared Error (RMSE) karena dapat mengukur seberapa jauh prediksi yang dihasilkan oleh model dari nilai yang sebenarnya. 

# Data Understanding
Dataset yang digunakan dalam proyek ini bersumber dari Kaggle dan berfokus pada destinasi wisata di Indonesia. Dataset ini terdiri dari beberapa kolom yang berisi informasi tentang tempat-tempat wisata di lima kota besar Indonesia, yaitu Jakarta, Yogyakarta, Semarang, Bandung, dan Surabaya. Data ini mencakup berbagai jenis destinasi, mulai dari wisata alam, budaya, hingga hiburan yang ada di kota-kota tersebut. Dataset ini dapat digunakan untuk menganalisis potensi wisata dan memberikan rekomendasi destinasi berdasarkan karakteristik yang ada. Dataset ini dapat diunduh dari Kaggle melalui informasi berikut:

**Judul**: [Indonesia Tourism Destination]

**Pemilik**: [A. Prabowo]

**Sumber**: Kaggle

Data ini terdiri dari 4 file, diantaranya
1. package_tourism.csv
2. tourism_rating.csv
3. tourism_with_id.csv
4. user.csv

Dari keempat file yang ada dalam dataset, satu file tidak akan digunakan dalam proses rekomendasi (package_tourism.csv), sehingga pada proyek ini hanya menggunakan 3 file datastet.

**Variabel Rating**
|No | Fitur | Keterangan |
| ------ | ------ | ------ |
| 1 | User_Id | ID unik untuk setiap pengguna yang memberikan rating
| 2 | Place_Id | ID  unik untuk setiap destinasi wisata yang ada dalam dataset
| 3 | Place_Ratings | Rating yang diberikan oleh pengguna untuk destinasi wisata yang telah dikunjungi

**Variabel Tempat Wisata**
|No | Fitur | Keterangan |
| ------ | ------ | ------ |
| 1 | Place_Id | ID unik untuk setiap destinasi wisata
| 2 | Place_Name | Nama dari destinasi wisata
| 3 | Description | Deskripsi singkat mengenai destinasi wisata
| 4 | Category | Kategori destinasi (misalnya alam, budaya, hiburan)
| 5 | City | Kota tempat destinasi wisata berada
| 6 | Price | Rentang harga yang diperlukan untuk mengunjungi destinasi
| 7 | Rating | Rata-rata nilai rating dari yang diberikan pengunjung
| 8 | Time_Minutes | Waktu yang dibutuhkan untuk mengunjungi destinasi dalam menit
| 9 | Coordinate |	Koordinat geografis destinasi dalam format (latitude, longitude).
| 10 | Lat | Nilai lintang (latitude) destinasi wisata.
| 11 | Long | Nilai bujur (longitude) destinasi wisata.

**Variabel Wisatawan**
|No | Fitur | Keterangan |
| ------ | ------ | ------ |
| 1 | User_Id | ID unik setiap pengguna untuk membedakan satu pengguna dengan pengguna lainnya
| 2 | Location | Lokasi atau kota tempat pengguna berada
| 3 | Age | Usia pengguna

## Data Loading
Pada tahap ini, dataset yang sebelumnya diunggah ke Google Drive diimpor untuk dianalisis lebih lanjut. Data kemudian ditampilkan 5 baris teratas menggunakan fungsi `head()`, hal ini bertujuan untuk memberikan gambaran awal tentang struktur dan isi dataset, seperti kolom yang ada dan apakah ada nilai yang hilang atau tidak sesuai. Langkah ini dilakukan agar dapat mengidentifikasi masalah pada data yang perlu ditangani lebih lanjut, seperti data yang tidak relevan sebelum melakukan analisis lebih mendalam. Dengan cara ini, data dapat dipersiapkan dengan baik untuk analisis lebih lanjut. Berikut adalah tampilan data secara garis besar dari masing masing dataset.

**Dataset Wisatawan**

<img src="https://github.com/user-attachments/assets/5806046d-bb94-43f9-a7ac-977c0949af55" alt="Load User" width="315">


**Dataset Tempat Wisata**

![Load Place](https://github.com/user-attachments/assets/be04e3e6-036c-4617-bf98-f30c39e00f80)


**Dataset Rating**

<img src="https://github.com/user-attachments/assets/c91c5e96-476b-40a5-bcf8-cb5e08f48d12" alt="Load Rating" width="285">


## Deskripsi Dataset
Deskripsi ini  memberikan pemahaman tentang setiap kolom dalam dataset, termasuk jenis data, satuan pengukuran, dan peran masing-masing variabel. Pada tahap ini, dijelaskan variabel agar analisis data berikutnya dapat dilakukan dengan tepat.

**Data Wisatawan**
|No | Kolom | Jumlah | Tipe Data |
| ------ | ------ | ------ |  ------ |
| 1   | User_Id  | 300 | int64 
| 2   | Location | 300 | object 
| 3   | Age      | 300 | int64

Data memiliki 300 baris untuk setiap kolom yang mana setiap fitur memiliki data dengan tipe numerik. Jumlah data pada masing-masing fitur tidak menunjukkan adanya missing value dalam dataset ini.

<img src="https://github.com/user-attachments/assets/5f65fb71-6f03-4aaf-b4f3-f8ede8153c38" alt="Describe User" width="235">

Ringkasan deskriptif di atas menunjukkan rentang usia wisatawan dari 18-40 tahun yang mana sekitar 50% wisatawan berusia dibawah 30 tahun. Selain itu, data ini juga mencakup lokasi wisatawan , tercatat wisatawan pada dataset ini berasal dari 28 kota/kabupaten. Jumlah ini diperoleh dengan menggunakan fungsi `len()` dan `unique()` seperti yang ditunjukkan di bawah ini.
```
print('Jumlah data lokasi pengguna: ', len(df_user.Location.unique()))
```
    Output
    Jumlah data lokasi pengguna:  28

**Data Tempat Wisata**

|No | Kolom | Jumlah | Tipe Data |
| ------ | ------ | ------ |  ------ |
| 1   | Place_Id     | 437 | int64 
| 2   | Place_Name   | 437 | object 
| 3   | Description  | 437 | object
| 4   | Category     | 437 | object 
| 5   | City         | 437 | object
| 6   | Price        | 437 | int64
| 7   | Rating       | 437 | float64 
| 8   | Time_Minutes | 205 | float64 
| 9   | Coordinate   | 437 | object 
| 10  | Lat          | 437 | float64 
| 11  | Long         | 437 | float64 

Data berjumlah 437 entri untuk setiap kolom, kecuali untuk kolom "Time_Minutes" yang memiliki 205 data, hal ini menunjukkan bahwa sebagian besar data terisi lengkap. Jenis data dalam kolom ini bervariasi, lebih dari 50% kolom berjenis numerik dengan tipe data `float64` yang mendominasi dan sisanya adalah kategorik berupa `object`.

<img src="https://github.com/user-attachments/assets/d1a284f0-b8f9-46c1-8b24-dbe4595a82ea" alt="Describe Place" width="800">

Data menunjukkan bahwa terdapat 437 tempat wisata dengan harga yang sangat beragam mulai dari gratis hingga Rp900.000 yang mana sebagian besar HTM untuk wisata ini adalah Rp5.000. Ini menunjukkan bahwa destinasi yang tercantum dalam dataset ini terbilang cukup terjangkau. Selain itu, destinasi terdata memiliki rating 3.4-5.0. Destinasi ini berasal dari 6 kota besar yang ada di Pulau Jawa, seperti Jakarta, Bandung, Yogyakarta, Semarang, dan Surabaya yang masing-masing dari destinasi tersebut dikategorikan dalam 5 jenis, yakni Budaya Taman Hiburan, Cagar Alam, Bahari, Pusat Perbelanjaan, dan Tempat Ibadah. Jumlah ini diperoleh melalui kode berikut.

```
print('Jumlah data tempat wisata: ', len(df_place.Place_Id.unique()), '\n')

print('Jumlah data kota dari tempat wisata: ', len(df_place.City.unique()))
print('Kota lokasi wisata: ', df_place.City.unique(), '\n')

print('Banyak kategori wisata: ', len(df_place.Category.unique()))
print('Kategori wisata: ', df_place.Category.unique())
```
    Output
    Jumlah data tempat wisata:  437 

    Jumlah data kota dari tempat wisata:  5
    Kota lokasi wisata:  ['Jakarta' 'Yogyakarta' 'Bandung' 'Semarang' 'Surabaya'] 

    Banyak kategori wisata:  6
    Kategori wisata:  ['Budaya' 'Taman Hiburan' 'Cagar  Alam' 'Bahari' 'Pusat Perbelanjaan' 'Tempat Ibadah']

**Data Rating**
|No | Kolom | Jumlah | Tipe Data |
| ------ | ------ | ------ |  ------ |
| 1   | User_Id       | 10000 | int64 
| 2   | Place_Id      | 10000 | int64 
| 3   | Place_Ratings | 10000 | int64

Mencakup 10000 data rating yang diberikan oleh wisatawan setelah mengunjungi destinasi tertentu dan keseluruhan data berjenis numerik dengan tipe data `int64`. Selain itu, dataset ini menunjukkan sebanyak 300 wisatawan memberi nilai untuk 437 destinasi wisata dengan rentang nilai 1-5 seperti pada gambar di bawah ini. 

<img src="https://github.com/user-attachments/assets/a711bb2f-905d-406d-b49c-389bac327a0c" alt="Describe Rating" width="335">

# Data Preparation

Data preparation dilakukan untuk memastikan model machine learning tidak mengalami overfitting dan dapat menggeneralisasi dengan baik. Tahap ini memungkinkan model dilatih menggunakan data latih dan dievaluasi menggunakan data uji yang belum pernah dilihat sebelumnya, sehingga memberikan gambaran tentang kinerja model. Selain itu, tahap ini juga memudahkan dalam pemisahan fitur (X) dan target (y), yang penting untuk pelatihan model yang efektif dan evaluasi yang akurat.

Pada tahap preparation ini dilakukan beberapa tahapan seperti preprocessing yang terdiri dari pengecekan missing value dan menyamakan data tempat wisata, TF-IDF Vectorizer, dan Train-Test-Split.Pada tahap ini, tidak dilakukan encoding karena data yang akan digunakan sudah bertipe numerik.

## Data Preprocessing
### Menghapus Missing Value
Penghapusan missing value bertujuan untuk memastikan kualitas data yang digunakan dalam analisis atau model. Missing value dapat menyebabkan ketidaktepatan hasil, terutama pada algoritma yang sensitif terhadap data yang tidak lengkap. Dengan menghapus missing value, bisa dipastikan bahwa data yang digunakan tidak mengganggu proses analisis lebih lanjut.

Pada tahap sebelumnya sudah dipastikan bahwa hampir seluruh data tidak memiliki missing value, hanya pada variabel 'Time_Minutes' terdapat missing value. Namun untuk memastikan kembali bahwa variabel yang akan digunakan benar-benar bersih dari missing value.

**Data Wisatawan**

<img src="https://github.com/user-attachments/assets/e606c56f-1f1f-471b-861e-7043edafe903" alt="Isnull User" width="175">

**Data Tempat Wisata**

<img src="https://github.com/user-attachments/assets/fc7f6eac-4bd0-4dd4-86c5-773083cc52fd" alt="Isnull Place" width="175">

**Data Rating**

<img src="https://github.com/user-attachments/assets/ab3c8007-7c71-4add-95f7-c486ddfae6f4" alt="Isnull Rating" width="175">

Hasil pengecekan di atas menunjukkan bahwa missing value hanya terdapat pada data tempat wisata seperti pada 'Time_Minutes' dan 'Unnamed: 11'. Karena nantinya akan dibuat dataframe baru, maka untuk missing value ini diabaikan termasuk kolom yang tidak relevan seperti 'Unnamed: 11' dan 'Unnamed: 12'.

### Menyamakan Data Tempat Wisata
Pada tahap ini, dilakukan beberapa hal, diantaranya.
1. Data `df_place` akan diurutkan terlebih dahulu berdasarkan 'Place_Id' lalu dibuatkan dataframe baru dengan nama inis`tempat_wisata` untuk data yang telah diurutkan. 
2. Data series pada Place_Id dan Place_Name akan diubah kedalam bentuk list. 
3. Data dibuatkan direktori baru bernama `wisata` untuk menampung variabel yang akan digunakan pada modeling, seperti 'id', 'nama_tempat', 'kategori', dan 'kota' seperti di bawah ini.

<img src="https://github.com/user-attachments/assets/fd51707a-e218-485c-92f2-566073c3dfcc" alt="Wisata" width="500">

## TF-IDF Vectorizer
TF-IDF Vectorizer digunakan untuk mengubah teks menjadi representasi numerik yang dapat digunakan dalam analisis lebih lanjut. Dengan menggabungkan frekuensi kemunculan kata (Term Frequency) dan pentingnya kata tersebut dalam konteks dokumen lainnya (Inverse Document Frequency), TF-IDF memberikan bobot pada kata-kata yang relevan dan jarang muncul di seluruh dataset. Ini meningkatkan kualitas pemodelan teks, mengurangi pengaruh kata-kata umum yang tidak memberi banyak informasi. Beberapa langkah yang dilakukan pada tahap ini diantaranya.
1. Melakukan inisialisasi TF-IDF Vectorizer dengan embuat objek TfidfVectorizer.
2. Menggabungkan dua kolom (kategori dan kota) dalam dataset dan mempersiapkan data untuk pemrosesan.
3. Menghitung IDF dan TF untuk kata-kata dalam kombinasi kolom 'kategori' dan 'kota'.
4. Menghasilkan vektor tf-idf dalam bentuk matriks dengan fungsi `todense()`.
5. Membuat dataframe untuk melihat tf-idf matrix dengan mengisi kolom dengan kategori dan kota sedangkan baris diisi dengan nama tempat.

Berikut adalah dataframe yang sudah dilakukan TF-IDF Vectorizer.

![Dataframe TF-IDF Vectorizer](https://github.com/user-attachments/assets/1972ac7c-1ca6-4a73-8896-f537b7fc3da5)


## Membagi Data untuk Training dan Validasi
Dataset dibagi menjadi dua bagian utama, yaitu data training dan data validasi untuk memastikan model yang dikembangkan dapat belajar dari data latih dan diukur kinerjanya menggunakan data validasi. Data latih bertujuan untuk mengajari model pola yang terdapat dalam data, sedangkan data validasi digunakan untuk mengevaluasi sejauh mana model dapat menggeneralisasi pola tersebut pada data baru yang belum pernah dilihat sebelumnya. Berikut adalah beberapaproses yang dilakukan pada tahap ini.
1. Mengecek jumlah data pada dataset rating seperti user dan tempat unik serta mengubah nilai rating menjadi float.
2. Menagacak data untuk memastikan bahwa data yang dihasilkan memiliki distribusi yang acak dan representatif.
3. Data train dan validasi dibagi dengan komposisi 80:20 dan juga memetakan data user dan tempat menjadi satu value terlebih dahulu serta mengubah rating menjadi skala 0 sampai 1 agar mudah dalam melakukan proses training.

# Modelling
Untuk memberikan rekomendasi tempat wisata, model yang digunakan diantaranya
1. Model Development dengan Content Based Filtering
2. Model Development dengan Collaborative Filtering

## Content Based Filtering
Model Development dengan Content-Based Filtering adalah proses membangun sistem rekomendasi yang merekomendasikan item kepada pengguna berdasarkan kesamaan fitur antara item yang sudah disukai pengguna dengan item lain dalam hal ini adalah tempat wisata. Proses ini melibatkan:
- Mengumpulkan data item atau tempat yang akan dianalisis, misalnya deskripsi, kategori, dan atribut lain yang relevan.
- Mengonversi data teks  menjadi vektor numerik menggunakan metode seperti TF-IDF.
- Menggunakan metrik seperti cosine similarity untuk mengukur kesamaan antar item berdasarkan vektor yang dihasilkan.
- Berdasarkan hasil perhitungan kesamaan, model memberikan rekomendasi item yang paling mirip dengan item yang disukai atau dipilih oleh pengguna.

|Kelebihan | Kekurangan |
| ------ | ------ |
| - Tidak memerlukan data pengguna sebelumnya, karena rekomendasi berdasarkan atribut konten. | - Hanya merekomendasikan item yang mirip dengan yang sudah ada, sehingga kurang variatif.
| - Dapat memberikan rekomendasi yang relevan meskipun pengguna baru atau data pengguna terbatas. | - Rentan terhadap masalah "filter bubble", di mana pengguna hanya mendapatkan rekomendasi yang terlalu mirip dan terbatas pada preferensi mereka.

Pada kasus ini, Content-Based Filtering bekerja dengan memanfaatkan konten tempat wisata seperti kategori dan kota untuk menentukan kesamaan antara tempat dengan menghitung cosine similarity antara vektor fitur, sistem bisa merekomendasikan tempat yang serupa dengan pilihan pengguna sebelumnya. TF-IDF Vectorizer membantu mengubah data teks menjadi representasi numerik sehingga perhitungan kesamaan bisa dilakukan menggunakan metode seperti cosine similarity.


### Cosine Similarity
Cosine Similarity adalah metrik yang digunakan untuk mengukur kesamaan antara dua vektor dalam ruang berdimensi tinggi berdasarkan sudut kosinus di antara keduanya. Nilainya berkisar antara -1 hingga 1, di mana 1 berarti kedua vektor sangat mirip, 0 berarti tidak ada hubungan, dan -1 berarti berlawanan sepenuhnya. Berikut adalah cara menggunakan cosine similarity untuk memberikan rekomendasi.
1. Data dikonversi menjadi vektor numerik, seperti menggunakan teknik TF-IDF Vectorizer seperti yang sudah dilakukan sebelumnya pada tahap Data Preparation
2. Menghitung hasil kali dot product antara dua vektor untuk mendapatkan angka yang menggambarkan hubungan antara keduanya.
3. Menghitung panjang (magnitudo) masing-masing vektor dengan menggunakan norma Euclidean, yaitu akar dari jumlah kuadrat nilai-nilai dalam vektor.
4. Setelahnya akan muncul interpretasi hasil. Nilai yang dihasilkan berkisar antara -1 hingga 1, di mana 1 berarti vektor sangat mirip, 0 berarti tidak ada kesamaan, dan -1 berarti berlawanan.

|Kelebihan | Kekurangan |
| ------ | ------ |
| - Mudah dipahami dan diimplementasikan. | - Tidak memperhitungkan magnitudo atau frekuensi absolut elemen.
| - Tidak terpengaruh oleh panjang vektor, hanya mempertimbangkan arah. | - Mungkin kurang efektif untuk data yang tidak terdistribusi dengan baik atau jika terdapat sedikit perbedaan antara item.
| - Cocok untuk data teks atau vektor sparse, seperti pada aplikasi NLP. | - Bisa menghasilkan hasil yang bias jika data tidak terstandarisasi dengan baik.

Berikut merupakan similarity matrix yang dihasilkan.

![Similarity Matrix](https://github.com/user-attachments/assets/8128c135-ee98-4591-8130-511020437693)

### Hasil Rekomendasi
Untuk membuat rekomendasi, dibuat sebuah fungsi bernama `tempat_rekomendasi`. Fungsi ini menggunakan matriks cosine similarity untuk menghitung tingkat kesamaan antara tempat yang dipilih dan tempat lainnya. Kemudian, tempat-tempat yang paling mirip (dengan jumlah yang ditentukan oleh parameter k) akan ditampilkan sebagai rekomendasi, beserta informasi tambahan seperti kategori dan kota tempat wisata tersebut. Dengan metode `argpartition`, fungsi ini menemukan indeks tempat yang memiliki skor kesamaan tertinggi. Kemudian, tempat yang sedang dicari dihapus dari daftar untuk memastikan hasil rekomendasi hanya mencakup tempat lain yang relevan. Pada kasus ini, tempat wisata yang ingin direkomendasikan adalah seperti wisata 'Surabaya North Quay'. Setelah itu, fungsi ini menyusun daftar rekomendasi yang paling mirip berdasarkan kesamaan, lengkap dengan informasi kategori dan kota untuk memudahkan pengguna dalam memilih tempat yang sesuai. Berikut adalah top 5 rekomendasi yang mirip dengan 'Surabaya North Quay'.

<img src="https://github.com/user-attachments/assets/74fbe8cd-a987-476c-8c49-6128944d7fa5" alt="Rekomendasi Content Based" width="350">

## Collaborative Filtering
Model Development dengan Collaborative Filtering adalah proses membangun sistem rekomendasi yang memanfaatkan pola interaksi pengguna terhadap item, seperti rating atau ulasan, untuk memberikan rekomendasi item kepada pengguna lain dengan preferensi serupa. Langkah-langkahnya meliputi:
- Embedding matriks digunakan untuk mewakili pengguna dan item dalam ruang berdimensi rendah.
- Model dilatih menggunakan data interaksi (rating atau preferensi) untuk mempelajari representasi embedding.
- Sistem menghitung kesamaan antara pengguna dan item menggunakan hasil embedding.
- Model memberikan rekomendasi item berdasarkan kesamaan pola interaksi antar pengguna.

|Kelebihan | Kekurangan |
| ------ | ------ |
|- Mampu menangkap preferensi kompleks tanpa memerlukan informasi eksplisit item. | - Membutuhkan banyak data interaksi untuk menghasilkan rekomendasi yang akurat.
|- Dapat merekomendasikan item baru kepada pengguna yang memiliki pola serupa dengan pengguna lain. | - Rentan terhadap masalah "cold start" untuk pengguna atau item baru yang tidak memiliki data interaksi.

Pada sistem rekomendasi tempat wisata, Collaborative Filtering memanfaatkan data interaksi seperti rating atau ulasan yang diberikan oleh pengguna terhadap tempat wisata. Dengan menggunakan embedding matriks, model dapat mempelajari hubungan antara pengguna dan tempat wisata, memungkinkan sistem merekomendasikan lokasi baru yang relevan berdasarkan pola interaksi pengguna lain dengan preferensi serupa. Metode ini efektif untuk menangkap preferensi pengguna tanpa bergantung pada informasi eksplisit tentang tempat wisata.

### RecommenderNet 
`RecommenderNet` adalah sebuah model rekomendasi berbasis jaringan saraf tiruan (Neural Network) yang menggunakan embedding untuk merepresentasikan pengguna dan item (misalnya tempat wisata). Model ini dirancang untuk menangkap pola hubungan antara pengguna dan item melalui interaksi mereka, memungkinkan prediksi relevansi atau peringkat suatu item. Berikut adalah cara kerja dari RecommenderNet.
1. Model menggunakan lapisan embedding untuk merepresentasikan pengguna dan item dalam vektor berdimensi rendah.
2. Dot product antara vektor embedding pengguna dan item dihitung untuk menghasilkan skor prediksi. Bias khusus untuk pengguna dan item ditambahkan.
3. Skor hasil ditransformasikan menggunakan fungsi sigmoid untuk menghasilkan probabilitas relevansi.
4. Model dilatih menggunakan binary crossentropy loss, dengan optimasi melalui algoritma Adam, dan evaluasi menggunakan metrik seperti RMSE.

|Kelebihan | Kekurangan |
| ------ | ------ |
| - Dapat menangkap pola non-linear antar pengguna dan item. | - Membutuhkan lebih banyak data dan waktu pelatihan.
| - Mengakomodasi bias pengguna dan item. | - Risiko overfitting jika data tidak mencukupi.
| - Fleksibel untuk berbagai jenis data. | - Memerlukan konfigurasi hyperparameter yang tepat.

### Hasil Rekomendasi
Fungsi ini menggunakan pendekatan Collaborative Filtering untuk memberikan rekomendasi tempat wisata berdasarkan rating pengguna sebelumnya. Dimulai dengan memilih pengguna secara acak, sistem kemudian mengambil daftar tempat yang sudah dikunjungi dan memberikan rekomendasi tempat yang belum dikunjungi namun memiliki prediksi rating tinggi.
Model ini menggunakan matriks prediksi untuk memproyeksikan rating yang mungkin diberikan pengguna terhadap tempat-tempat yang belum dikunjungi. Hasil prediksi ini kemudian diurutkan berdasarkan rating tertinggi dan ditampilkan sebagai rekomendasi. Berikut adalah top 10 rekomendasi tempat wisata menggunakan berdasarkan rating. 

<img src="https://github.com/user-attachments/assets/78d1a23a-860a-4580-a2e1-77896bc185e6" alt="Rekomendasi Collaborative" width="450">

# Evaluation
RMSE (Root Mean Squared Error) adalah metrik yang digunakan untuk mengukur seberapa besar perbedaan antara nilai prediksi dan nilai aktual dalam satuan yang sama dengan data asli. Proses perhitungannya melibatkan langkah pertama untuk mengkuadratkan selisih antara nilai prediksi dan nilai sebenarnya, lalu menghitung rata-rata dari hasil kuadrat tersebut, dan akhirnya mengambil akar kuadrat dari rata-rata tersebut. Hasil RMSE yang lebih rendah menunjukkan model yang lebih baik dalam memprediksi dengan akurat, karena kesalahan yang lebih kecil menghasilkan nilai RMSE yang lebih rendah.

**Formula RMSE**

<img src="https://github.com/user-attachments/assets/9737e1fa-2a4e-4b92-bbe8-673a6825f895" alt="Formula RMSE" width="300">

Keterangan
- $y_i$ : Nilai aktual
- $\hat{yi}$ : nilai yang diprediksi oleh model
- $n$ : jumlah data (jumlah sampel)

**Hasil Evaluasi Berdasarkan Metrik RMSE**

![Hasil Metrik RMSE](https://github.com/user-attachments/assets/a2839a62-36dd-420a-960b-f7fec95985e9)

Grafik yang ditampilkan menunjukkan RMSE untuk data train dan data test selama 100 epoch. Nilai RMSE untuk data train (garis biru) menurun secara konsisten dari sekitar 0.34 menjadi sekitar 0.33 menunjukkan bahwa model semakin baik dalam memprediksi data pelatihan seiring berjalannya waktu. Sementara itu, RMSE untuk data test (garis oranye) juga menunjukkan penurunan yang lebih terbatas di sekitar 0.34 setelah beberapa epoch.

# Kesimpulan
Penggunaan content-based filtering dan collaborative filtering telah menjawab problem statement yang mana model mampu memberikan rekomendasi destinasi berdasarkan karakteristik destinasi yang serupa dan rating wisatawan, dengan hasil yang relevan dan personal. Model berhasil menganalisis karakteristik destinasi serta data rating untuk memberikan rekomendasi yang lebih akurat sesuai dengan preferensi wisatawan. Dengan menggunakan metode seperti cosine similarity dan RMSE, model meningkatkan kualitas rekomendasi destinasi dengan memanfaatkan data yang ada untuk menyarankan destinasi yang sesuai dengan preferensi wisatawan.

Referensi

Kemenparekraf. (2024, Oktober 02). Kemenparekraf Promosikan Bangga Berwisata di Indonesia Lewat DIATF 2024. _Kemenparekraf/Baparekraf RI_. https://www.kemenparekraf.go.id/berita/siaran-pers-kemenparekraf-promosikan-bangga-berwisata-di-indonesia-lewat-diatf-2024

[Kemenparekraf, 2024]: <https://www.kemenparekraf.go.id/berita/siaran-pers-kemenparekraf-promosikan-bangga-berwisata-di-indonesia-lewat-diatf-2024>
[Indonesia Tourism Destination]: <https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination/data>
[A. Prabowo]: <https://www.kaggle.com/aprabowo>
