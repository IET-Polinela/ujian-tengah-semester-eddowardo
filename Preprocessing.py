from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 1. Salin data
df_clean = df.copy()

# 2. Hapus kolom ID
df_clean.drop(columns='id', inplace=True)

# 3. Ubah nilai 'Other' pada gender jadi mode
if 'Other' in df_clean['gender'].values:
    df_clean['gender'] = df_clean['gender'].replace('Other', df_clean['gender'].mode()[0])

# 4. Isi nilai kosong di 'bmi' dengan median
df_clean['bmi'] = df_clean['bmi'].fillna(df_clean['bmi'].median())

# 5. One-hot encoding fitur kategorikal
categorical = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_clean = pd.get_dummies(df_clean, columns=categorical, drop_first=True)

# 6. Normalisasi fitur numerik
scaler = MinMaxScaler()
numeric = ['age', 'avg_glucose_level', 'bmi']
df_clean[numeric] = scaler.fit_transform(df_clean[numeric])

# 7. Simpan data hasil encode
df_clean.to_csv('encoded_stroke_data.csv', index=False)
