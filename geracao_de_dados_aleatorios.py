import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

# Features
renda = np.random.randint(1200, 15000, n)
score = np.random.randint(400, 900, n)
estado_civil = np.random.choice(["solteiro", "casado", "divorciado"], n)
tempo_emprego = np.random.randint(0, 20, n)
patrimonio = renda * np.random.uniform(5, 20, n)
dividas = renda * np.random.uniform(0.5, 5, n)

# Criar "risco" com lógica imperfeita + ruído
risco = []

for i in range(n):
    prob = 0
    
    # fatores positivos
    if renda[i] > 6000:
        prob -= 1
    if score[i] > 700:
        prob -= 2
    if tempo_emprego[i] > 5:
        prob -= 1
    if patrimonio[i] > 80000:
        prob -= 1
    
    # fatores negativos
    if dividas[i] > renda[i] * 2:
        prob += 2
    if score[i] < 600:
        prob += 2
    if tempo_emprego[i] < 2:
        prob += 1
    
    # ruído aleatório (ESSENCIAL)
    prob += np.random.normal(0, 1.5)
    
    risco.append(1 if prob > 0 else 0)

# Criar DataFrame
df = pd.DataFrame({
    "renda_mensal": renda,
    "score_credito": score,
    "estado_civil": estado_civil,
    "tempo_emprego_anos": tempo_emprego,
    "patrimonio": patrimonio.astype(int),
    "dividas_pendentes": dividas.astype(int),
    "risco": risco
})

# Salvar CSV
df.to_csv("data.csv", index=False)

print("CSV gerado com sucesso!")
print(df["risco"].value_counts(normalize=True))