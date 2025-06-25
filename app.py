import streamlit as st
import pandas as pd
from markov_model import prediksi_markov
from ai_model import prediksi_ai

st.title("🎰 Prediksi Togel 4 Digit - AI & Markov (Live Retrain)")

# Input riwayat angka
riwayat_input = st.text_area("📝 Masukkan data history togel (1 angka per baris):", height=200)
data_lines = [line.strip() for line in riwayat_input.split("\n") if line.strip().isdigit() and len(line.strip()) == 4]
df = pd.DataFrame({"angka": data_lines})

# Input angka aktual
angka_aktual = st.text_input("❓ Masukkan angka aktual (untuk uji akurasi, opsional):", "")

# Jumlah data uji
jumlah_uji = st.number_input("📊 Jumlah data uji terakhir:", min_value=1, max_value=500, value=5, step=1)

# Pilih metode prediksi
metode = st.selectbox("🧠 Pilih Metode Prediksi", ["Markov", "LSTM AI"])

# Tombol prediksi
if st.button("🔮 Prediksi"):
    if metode == "Markov":
        hasil = prediksi_markov(df)
    else:
        hasil = prediksi_ai(df)

    st.success(f"🎯 Prediksi ({metode}): {hasil}")

    # Uji akurasi jika angka aktual dimasukkan
    if angka_aktual and angka_aktual.isdigit() and len(angka_aktual) == 4:
        uji_df = df.tail(jumlah_uji)
        total = 0
        benar = 0

        for i in range(jumlah_uji):
            subset_df = df.iloc[:-(jumlah_uji - i)]
            if len(subset_df) >= 11:
                if metode == "LSTM AI":
                    pred = prediksi_ai(subset_df)
                else:
                    pred = prediksi_markov(subset_df)

                actual = f"{int(uji_df.iloc[i]['angka']):04d}"
                match = sum([p == a for p, a in zip(pred, actual)])
                benar += match
                total += 4

        if total > 0:
            akurasi_total = (benar / total) * 100
            st.info(f"📈 Akurasi per digit (dari {jumlah_uji} data): {akurasi_total:.2f}%")
        else:
            st.warning("❌ Tidak cukup data untuk menghitung akurasi.")
