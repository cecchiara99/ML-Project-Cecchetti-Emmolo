import pandas as pd

# Specifica i percorsi dei tuoi file di addestramento e di test
percorso_file_train_1 = './monk+s+problems/monks-1.train'
percorso_file_train_2 = './monk+s+problems/monks-2.train'
percorso_file_train_3 = './monk+s+problems/monks-3.train'

# Carica i file di addestramento e di test in DataFrame separati
df_train_1 = pd.read_csv(percorso_file_train_1)
df_train_2 = pd.read_csv(percorso_file_train_2)
df_train_3 = pd.read_csv(percorso_file_train_3)

print(df_train_1.head(1))