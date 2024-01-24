from itertools import product

def generate_combinations(iperparametri):
    # Estrai i nomi e i valori degli iperparametri
    hyperparametri_nomi = list(iperparametri.keys())
    hyperparametri_valori = list(iperparametri.values())

    combinazioni_valori = list(product(*hyperparametri_valori))

    hyperparameters = []
    for combinazione in combinazioni_valori:
        dizionario_combinazione = dict(zip(hyperparametri_nomi, combinazione))
        hyperparameters.append(dizionario_combinazione)

    return hyperparameters

# Esempio di utilizzo
iperparametri = {
    'parametro1': [1, 2],
    'parametro2': ['a', 'b', 'c'],
    'parametro3': [True, False]
}

risultato = generate_combinations(iperparametri)

print(risultato)

# Stampa il risultato
for combinazione in risultato:
    print(combinazione)
