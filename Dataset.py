import pandas as pd

# Load data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Lihat info dasar
print("Jumlah observasi:", df.shape[0])
print("Jumlah fitur:", df.shape[1] - 1)  # Kecuali label stroke
df.describe()
