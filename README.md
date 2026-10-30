# Peramalan Harga Emas Harian dengan LSTM (TA-04)

Aplikasi web Flask yang menampilkan riwayat harga emas, evaluasi model LSTM, dan ramalan hari kerja berikutnya.
Mata kuliah Pembelajaran Mesin, Modul 04.

## Isi repositori
| Berkas | Fungsi |
|---|---|
| `train.py` / `analisis_lstm.ipynb` | EDA, scaling, windowing, model LSTM, evaluasi, simpan model dan scaler (isinya sama) |
| `app.py`, `templates/`, `static/` | Aplikasi Flask |
| `data/gold_price.csv` | Dataset (kolom `date`, `price`) |
| `model/` | Hasil pelatihan: `gold_lstm.h5`, `scaler.joblib`, dan file JSON hasil evaluasi |
| `requirements.txt`, `Procfile`, `render.yaml` | Konfigurasi deploy |

## 1. Latih model (wajib, sekali saja)
Folder `model/` sengaja kosong sampai kamu melatih modelnya.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # Mac Apple Silicon: ganti tensorflow-cpu dengan tensorflow
python train.py                      # beberapa menit; atau jalankan analisis_lstm.ipynb
```
Hasilnya: `model/gold_lstm.h5`, `model/scaler.joblib`, `metrics.json`, `eval.json`, `history.json`, `forecast.json`, `config.json`.
Latih di lingkungan dengan versi library yang sama dengan `requirements.txt` supaya scaler dan model terbaca di server.

## 2. Jalankan lokal
Buka lewat server, **jangan** klik dua kali `index.html`.
```bash
python app.py        # buka http://127.0.0.1:5000
```
Tanpa model, halaman riwayat harga tetap jalan, dan bagian akurasi serta ramalan menampilkan petunjuk melatih model.

## 3. Upload ke GitHub
```bash
git init && git add . && git commit -m "TA-04 peramalan harga emas LSTM"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git push -u origin main
```
Pastikan folder `model/` ikut ter-commit (ukurannya hanya beberapa ratus KB).

## 4. Deploy
**Render:** New, Blueprint (atau Web Service), pilih repo. `render.yaml` sudah berisi perintah build dan start.
**Railway:** New Project, Deploy from GitHub repo. `Procfile` dipakai otomatis.
TensorFlow cukup berat untuk paket gratis dengan RAM 512 MB. Kalau server kehabisan memori, aplikasi tetap menampilkan ramalan dari `forecast.json` (30 hari) selama model gagal dimuat.

## Catatan data
- Data dimulai dari 2000 (sebelumnya rezim harga sangat berbeda). Hari tanpa harga diisi `ffill`.
- Pembagian data 80/20 menurut urutan waktu, dan scaler hanya dilatih pada data latih.
- Dataset berakhir pada April 2021, jadi ramalan dihitung dari tanggal tersebut.
- Satuan harga di UI diatur lewat `PRICE_UNIT` di `app.py` (default `USD/oz`). Sesuaikan dengan sumber dataset.
