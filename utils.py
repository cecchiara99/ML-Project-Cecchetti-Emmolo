
"""
# Carica i file di addestramento e di test in DataFrame separati
df_train_1 = pd.read_csv(percorso_file_train_1)
df_train_2 = pd.read_csv(percorso_file_train_2)
df_train_3 = pd.read_csv(percorso_file_train_3)
"""

"""
# Leggi il dataset di addestramento 1
col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
monk_dataset = pd.read_csv(percorso_file_train_1, sep=' ', names=col_names)
monk_dataset.set_index('Id', inplace=True)
labels = monk_dataset.pop('class')

# Seleziona solo le colonne numeriche per la normalizzazione
numeric_columns = monk_dataset.columns
numeric_data = monk_dataset[numeric_columns]

# Normalizza manualmente le colonne numeriche
normalized_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())

# Riunisci il dataset normalizzato con le etichette
monk_dataset_normalized = pd.concat([normalized_data, labels], axis=1)

# Stampa il dataset normalizzato
print("Dataset normalizzato:")
print(monk_dataset_normalized)
"""