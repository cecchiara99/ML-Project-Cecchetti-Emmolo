import pandas as pd

# Specifica i percorsi dei tuoi file di addestramento e di test
percorso_file_train = './monk+s+problems/monks-1.train'
percorso_file_test = './monk+s+problems/monks-1.test'

# Carica i file di addestramento e di test in DataFrame separati
df_train = pd.read_csv(percorso_file_train)
df_test = pd.read_csv(percorso_file_test)

# Specifica il percorso del tuo file .names
percorso_file_names = './monk+s+problems/monks.names'

# Stampa i nomi delle colonne e la prima riga per il file di addestramento
print("Nomi delle colonne (train):", df_train.columns.tolist())
print("Prima riga (train):", df_train.iloc[0].tolist())

# Stampa i nomi delle colonne e la prima riga per il file di test
print("\nNomi delle colonne (test):", df_test.columns.tolist())
print("Prima riga (test):", df_test.iloc[0].tolist())
