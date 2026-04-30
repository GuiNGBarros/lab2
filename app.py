import streamlit as st
import joblib
import numpy as np


# ================================
# CARREGAMENTO DO MODELO
# ================================

# Carrega o modelo previamente treinado
modelo = joblib.load("modelo.pkl")


# ================================
# CONFIGURAÇÃO DA PÁGINA
# ================================

# Define configurações visuais da página
st.set_page_config(
    page_title="Classificação de Risco de Crédito",  # título da aba do navegador
    page_icon="💳",  # ícone da aba
    layout="wide"  # usa toda largura da tela
)


# ================================
# TÍTULO E DESCRIÇÃO
# ================================

# Título principal
st.title("💳 Classificação de Risco de Crédito")

# Pequena descrição do que o app faz
st.markdown("""
Preencha os dados abaixo para prever se um cliente é:

- ✅ **Bom Pagador (Aprovado)**  
- ⚠️ **Alto Risco de Inadimplência (Negado)**
""")


# ================================
# INPUTS DO USUÁRIO
# ================================

# Criamos colunas para melhorar a UI
col1, col2 = st.columns(2)


# -------- COLUNA 1 --------
with col1:
    # Input de renda mensal
    renda = st.number_input("Renda Mensal (R$)", min_value=0.0, value=3000.0)

    # Input de score de crédito
    score = st.number_input("Score de Crédito", min_value=0, max_value=1000, value=650)

    # Input de tempo no emprego
    tempo_emprego = st.number_input("Tempo no Emprego (anos)", min_value=0, value=2)


# -------- COLUNA 2 --------
with col2:
    # Input de patrimônio
    patrimonio = st.number_input("Patrimônio (R$)", min_value=0.0, value=20000.0)

    # Input de dívidas
    dividas = st.number_input("Dívidas Pendentes (R$)", min_value=0.0, value=5000.0)

    # Selectbox para estado civil (categoria)
    estado_civil = st.selectbox(
        "Estado Civil",
        ["solteiro", "casado", "divorciado"]
    )


# ================================
# ENCODING DO ESTADO CIVIL
# ================================

# Criamos manualmente o mesmo tipo de mapeamento que você fez no dataset
mapa_estado_civil = joblib.load("mapa_estado_civil.pkl")

# Converte a categoria para número
estado_civil_encoded = mapa_estado_civil[estado_civil]


# ================================
# BOTÃO DE PREVISÃO
# ================================

# Quando o botão for clicado
if st.button("🔍 Fazer Previsão"):

    # Monta o array na MESMA ORDEM das features usadas no treinamento
    # [renda, score, estado_civil, tempo_emprego, patrimonio, dividas]
    input_array = np.array([
        renda,
        score,
        estado_civil_encoded,
        tempo_emprego,
        patrimonio,
        dividas
    ]).reshape(1, -1)  # reshape para formato (1 linha, N colunas)


    # ================================
    # PREDIÇÃO DO MODELO
    # ================================

    # Classe prevista (0 ou 1)
    prediction = modelo.predict(input_array)[0]

    # Probabilidades (ex: [0.8, 0.2])
    prediction_proba = modelo.predict_proba(input_array)[0]


    # ================================
    # TRADUÇÃO DO RESULTADO
    # ================================

    # Converte saída numérica em texto amigável
    if prediction == 0:
        resultado = "✅ Bom Pagador / Aprovado"
    else:
        resultado = "⚠️ Alto Risco de Inadimplência / Negado"


    # ================================
    # EXIBIÇÃO DOS RESULTADOS
    # ================================

    st.subheader("Resultado da Análise")

    # Mostra a classe prevista
    st.write(f"**Classificação:** {resultado}")

    # Mostra probabilidades de forma mais clara
    st.write(f"**Probabilidade de Bom Pagador:** {prediction_proba[0]:.2%}")
    st.write(f"**Probabilidade de Inadimplência:** {prediction_proba[1]:.2%}")


    # ================================
    # INFORMAÇÃO EXTRA (DEBUG / TRANSPARÊNCIA)
    # ================================

    # Mostra os dados usados na previsão (útil para debug)
    with st.expander("Ver dados enviados ao modelo"):
        st.write(input_array)