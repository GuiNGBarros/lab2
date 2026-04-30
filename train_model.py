
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ================================
# CARREGAMENTO DO DATASET
# ================================

# Lê o arquivo CSV contendo os dados
df = pd.read_csv("data.csv")


# ================================
# ENCODING DA VARIÁVEL CATEGÓRICA
# ================================

# Cria um dicionário que mapeia cada categoria para um número
# Exemplo: {"solteiro": 0, "casado": 1, "divorciado": 2}
mapa = {
    "solteiro": 0,
    "casado": 1,
    "divorciado": 2
}

# Aplica o mapeamento na coluna "estado_civil"
# Substitui texto por números
df["estado_civil"] = df["estado_civil"].map(mapa)

# (Boa prática) Exibir o mapeamento para garantir consistência depois
print("Mapeamento estado_civil:", mapa)


# ================================
# SEPARAÇÃO DE FEATURES (X) E TARGET (y)
# ================================

# Remove a coluna "risco" e usa o restante como variáveis independentes (features)
X = df.drop(columns=["risco"]).values

# Seleciona a coluna "risco" como variável alvo (target)
y = df["risco"].values


# ================================
# DIVISÃO TREINO / TESTE
# ================================

# Divide os dados:
# - 80% para treino
# - 20% para teste
# random_state garante reprodutibilidade (mesmo resultado sempre)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ================================
# CRIAÇÃO E TREINAMENTO DO MODELO
# ================================

# Instancia o modelo Random Forest
# n_estimators = número de árvores (quanto maior, mais robusto, porém mais lento)
# random_state = garante resultados consistentes
modelo = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Treina o modelo com os dados de treino
modelo.fit(X_train, y_train)


# ================================
# PREDIÇÃO
# ================================

# Faz previsões usando os dados de teste
# CORREÇÃO: variável estava escrita "predicions"
predictions = modelo.predict(X_test)


# ================================
# AVALIAÇÃO DO MODELO
# ================================

# Calcula a acurácia comparando valores reais vs previstos
accuracy = accuracy_score(y_test, predictions)

# Exibe a acurácia formatada com 2 casas decimais
print(f"Acurácia: {accuracy:.2f}")


# ================================
# SALVANDO O MODELO
# ================================

# Salva o modelo treinado em um arquivo .pkl
# Esse arquivo será usado depois no app (Streamlit)
joblib.dump(modelo, "modelo.pkl")


# ================================
# (OPCIONAL MAS MUITO IMPORTANTE)
# SALVAR O MAPEAMENTO
# ================================

# Salvar o dicionário de encoding também é essencial
# para garantir que o app use o mesmo padrão
joblib.dump(mapa, "mapa_estado_civil.pkl")