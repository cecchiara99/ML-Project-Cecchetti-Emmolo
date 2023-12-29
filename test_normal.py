import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import anderson
import numpy as np
from statsmodels.stats.diagnostic import lilliefors


# Specifica i percorsi dei tuoi file di addestramento e di test
percorso_file_train_1 = './monk+s+problems/monks-1.train'
percorso_file_train_2 = './monk+s+problems/monks-2.train'
percorso_file_train_3 = './monk+s+problems/monks-3.train'

# Carica i file di addestramento e di test in DataFrame separati
df_train_1 = pd.read_csv(percorso_file_train_1)
df_train_2 = pd.read_csv(percorso_file_train_2)
df_train_3 = pd.read_csv(percorso_file_train_3)

print(df_train_1.head(1))

data_1 = []
data_2 = []
data_3 = []

datasets = [df_train_1, df_train_2, df_train_3]

for dataset in datasets:
    for index, row in dataset.iterrows():
        # Dividi la stringa in una lista di valori
        record_values = [int(value) if value.isdigit() else value for value in ' '.join(row).split()]
        # Rimuovi l'ultimo valore dalla lista
        record_values.pop()
        if dataset.equals(df_train_1):
            data_1.append(record_values)
        elif dataset.equals(df_train_2):
            data_2.append(record_values)
        else:
            data_3.append(record_values)

print("\nDATA_1:\n")
print(data_1)
print("\n")

print("\nDATA_2:\n")
print(data_2)
print("\n")

print("\nDATA_3:\n")
print(data_3)
print("\n")
    
# Flatten the list of lists into a single list
all_values_1 = [value for sublist in data_1 for value in sublist]
all_values_2 = [value for sublist in data_2 for value in sublist]
all_values_3 = [value for sublist in data_3 for value in sublist]

# Perform the Shapiro-Wilk test for normality on the combined data
significance_level = 0.05
print("\nSignificance Level: ",significance_level,"\n")
stat_1, p_value_1 = stats.shapiro(all_values_1)


print(f"Shapiro-Wilk test statistic: {stat_1}, p-value: {p_value_1}")


if p_value_1 > significance_level:
    print("Globalmente, i dati sembrano seguire una distribuzione normale (MONK1).")
else:
    print("Globalmente, i dati non sembrano seguire una distribuzione normale (MONK1).\n")

stat_2, p_value_2 = stats.shapiro(all_values_2)
print(f"Shapiro-Wilk test statistic: {stat_2}, p-value: {p_value_2}")

if p_value_2 > significance_level:
    print("Globalmente, i dati sembrano seguire una distribuzione normale (MONK2).")
else:
    print("Globalmente, i dati non sembrano seguire una distribuzione normale (MONK2).\n")


stat_3, p_value_3 = stats.shapiro(all_values_3)
print(f"Shapiro-Wilk test statistic: {stat_3}, p-value: {p_value_3}")
if p_value_3 > significance_level:
    print("Globalmente, i dati sembrano seguire una distribuzione normale (MONK3).")
else:
    print("Globalmente, i dati non sembrano seguire una distribuzione normale (MONK3).\n")


# Function to determine whether to reject the null hypothesis for Anderson-Darling test
def reject_null_hypothesis_anderson(test_statistic, critical_values, significance_levels):
    for i, level in enumerate(significance_levels):
        if test_statistic > critical_values[i]:
            print(f"At {level}% significance level: Reject the null hypothesis (Anderson-Darling test)\n")
        else:
            print(f"At {level}% significance level: Fail to reject the null hypothesis (Anderson-Darling test)\n")

# Anderson-Darling test for each dataset
result_1 = anderson(all_values_1)
print(f"\nAnderson-Darling test statistic 1: {result_1.statistic}, critical values: {result_1.critical_values}, significance level: {result_1.significance_level}")
reject_null_hypothesis_anderson(result_1.statistic, result_1.critical_values, result_1.significance_level)

result_2 = anderson(all_values_2)
print(f"\nAnderson-Darling test statistic 2: {result_2.statistic}, critical values: {result_2.critical_values}, significance level: {result_2.significance_level}")
reject_null_hypothesis_anderson(result_2.statistic, result_2.critical_values, result_2.significance_level)

result_3 = anderson(all_values_3)
print(f"\nAnderson-Darling test statistic 3: {result_3.statistic}, critical values: {result_3.critical_values}, significance level: {result_3.significance_level}")
reject_null_hypothesis_anderson(result_3.statistic, result_3.critical_values, result_3.significance_level)


from scipy.stats import kstest

# Set the significance level
alpha = 0.05

# Kolmogorov-Smirnov Tests
result_1 = kstest(all_values_1, 'norm')
print(f"Kolmogorov-Smirnov test statistic 1: {result_1.statistic}, p-value: {result_1.pvalue}")
if result_1.pvalue > alpha:
    print("Distribution 1 appears to be normal.")
else:
    print("Distribution 1 does not appear to be normal.\n")

result_2 = kstest(all_values_2, 'norm')
print(f"Kolmogorov-Smirnov test statistic 2: {result_2.statistic}, p-value: {result_2.pvalue}")
if result_2.pvalue > alpha:
    print("Distribution 2 appears to be normal.")
else:
    print("Distribution 2 does not appear to be normal.\n")

result_3 = kstest(all_values_3, 'norm')
print(f"Kolmogorov-Smirnov test statistic 3: {result_3.statistic}, p-value: {result_3.pvalue}")
if result_3.pvalue > alpha:
    print("Distribution 3 appears to be normal.")
else:
    print("Distribution 3 does not appear to be normal.")
print("\n")

# Lilliefors Tests
result_1, p_value_1 = lilliefors(all_values_1)
print(f"Lilliefors test statistic 1: {result_1}, p-value: {p_value_1}")
if p_value_1 > alpha:
    print("Distribution 1 appears to be normal.")
else:
    print("Distribution 1 does not appear to be normal.\n")

result_2, p_value_2 = lilliefors(all_values_2)
print(f"Lilliefors test statistic 2: {result_2}, p-value: {p_value_2}")
if p_value_2 > alpha:
    print("Distribution 2 appears to be normal.")
else:
    print("Distribution 2 does not appear to be normal.\n")

result_3, p_value_3 = lilliefors(all_values_3)
print(f"Lilliefors test statistic 3: {result_3}, p-value: {p_value_3}")
if p_value_3 > alpha:
    print("Distribution 3 appears to be normal.")
else:
    print("Distribution 3 does not appear to be normal.\n")
print("\n")






