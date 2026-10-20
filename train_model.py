import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

def main():
    print("=" * 60)
    print("      PROSES PELATIHAN MODEL REGRESI LINEAR SEDERHANA")
    print("=" * 60)

    # 1. Pastikan folder penyimpan tersedia
    os.makedirs('models', exist_ok=True)
    dataset_path = os.path.join('dataset', 'pseudo_facebook.csv')

    if not os.path.exists(dataset_path):
        print(f"[ERROR] File dataset tidak ditemukan di '{dataset_path}'!")
        return

    # 2. Load Dataset
    print("[1/5] Membaca dataset...")
    df = pd.read_csv(dataset_path)
    print(f"      Total data mentah: {len(df):,} baris")

    # 3. Preprocessing Data
    # Fitur X = likes (diberikan), Target y = likes_received (diterima)
    df_clean = df[['likes', 'likes_received']].dropna()
    
    # Hapus outlier ekstrem untuk menjaga stabilitas model regresi
    q_x = df_clean['likes'].quantile(0.99)
    q_y = df_clean['likes_received'].quantile(0.99)
    df_filtered = df_clean[(df_clean['likes'] <= q_x) & (df_clean['likes_received'] <= q_y)]
    print(f"      Total data setelah pembersihan outlier: {len(df_filtered):,} baris")

    X = df_filtered[['likes']]
    y = df_filtered['likes_received']

    # 4. Split Data Training & Testing (80:20)
    print("[2/5] Membagi data train (80%) dan test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Pelatihan Model Regresi Linear
    print("[3/5] Melatih model Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluasi Prediksi
    y_pred = model.predict(X_test)

    # Hitung Metrik Statistik
    intercept = float(model.intercept_)
    slope = float(model.coef_[0])
    mae = float(metrics.mean_absolute_error(y_test, y_pred))
    mse = float(metrics.mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(metrics.r2_score(y_test, y_pred))
    corr = float(df_filtered['likes'].corr(df_filtered['likes_received']))

    print("\n--- HASIL EVALUASI MODEL ---")
    print(f"Intercept (Beta 0) : {intercept:.4f}")
    print(f"Slope (Beta 1)     : {slope:.4f}")
    print(f"Correlation (r)    : {corr:.4f}")
    print(f"MAE                : {mae:.4f}")
    print(f"MSE                : {mse:.4f}")
    print(f"RMSE               : {rmse:.4f}")
    print(f"R-Squared (R2)     : {r2:.4f}")

    # Save Model ke folder models/
    model_save_path = os.path.join('models', 'model_likes.joblib')
    joblib.dump(model, model_save_path)
    print(f"\n[4/5] Model berhasil disimpan ke '{model_save_path}'")

    # Save Ringkasan Metrik JSON untuk dibaca oleh app.py
    metrics_data = {
        "intercept": intercept,
        "slope": slope,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "corr": corr,
        "dataset_count": len(df_filtered),
        "mean_likes_given": float(df_filtered['likes'].mean()),
        "mean_likes_received": float(df_filtered['likes_received'].mean())
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)

    # 6. Pembuatan Gambar Grafik (Figure_1.png dan Figure_2.png)
    print("[5/5] Membuat grafik analisis data...")

    # Figure 1: Scatterplot + Best-Fit Linear Regression Line
    plt.figure(figsize=(9, 6), dpi=150)
    sns.set_theme(style="whitegrid")
    
    # Sample 1000 titik agar visualisasi jernih
    sample_df = df_filtered.sample(n=min(1000, len(df_filtered)), random_state=42)
    plt.scatter(sample_df['likes'], sample_df['likes_received'], alpha=0.35, color='#0d6efd', label='Data Sampel Observasi')
    
    # Plot Garis Regresi Best-Fit
    x_range = np.linspace(0, sample_df['likes'].max(), 100)
    y_line = intercept + slope * x_range
    plt.plot(x_range, y_line, color='#dc3545', linewidth=2.5, label=f'Garis Regresi ($y = {intercept:.2f} + {slope:.2f}X$)')
    
    plt.title("Figure 1: Hubungan Regresi Likes Diberikan vs Likes Diterima", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Likes Diberikan ke Orang Lain (X)", fontsize=10)
    plt.ylabel("Likes Diterima dari Orang Lain (y)", fontsize=10)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('Figure_1.png')
    plt.close()

    # Figure 2: Residual Plot (Distribusi Error Prediksi)
    plt.figure(figsize=(9, 6), dpi=150)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, color='#0dcaf0', bins=40)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
    plt.title("Figure 2: Distribusi Error Prediksi (Residuals)", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Selisih Nilai Aktual - Nilai Prediksi (Residual)", fontsize=10)
    plt.ylabel("Frekuensi", fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Figure_2.png')
    plt.close()

    print("      'Figure_1.png' dan 'Figure_2.png' berhasil diperbarui!")
    print("=" * 60)
    print("  PELATIHAN SELESAI! SIAP DIJALANKAN DI FLASK (app.py)")
    print("=" * 60)

if __name__ == '__main__':
    main()