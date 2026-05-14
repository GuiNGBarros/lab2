Pipeline de Machine Learning – Risco de Crédito (Streamlit + Docker)

Este projeto implementa um pipeline completo de machine learning para classificação de risco de crédito, abrangendo desde a geração sintética dos dados até o deploy de uma interface web interativa em Streamlit, totalmente containerizada com Docker.

Pipeline de ML
O fluxo foi estruturado em etapas modulares:

🔹 1. Geração de dados sintéticos
Criação de um dataset com 1.000 registros simulando perfis de clientes
Features: renda mensal, score de crédito, estado civil, tempo de emprego, patrimônio e dívidas
Geração da variável alvo risco a partir de regras heurísticas combinadas com ruído aleatório, evitando padrões triviais
Exportação para data.csv
🔹 2. Treinamento do modelo
Pré-processamento com encoding manual da variável categórica estado_civil
Split treino/teste (80/20) com random_state fixo para reprodutibilidade
Treinamento de um RandomForestClassifier com 100 árvores
Avaliação por acurácia no conjunto de teste
Persistência do modelo (modelo.pkl) e do mapeamento de encoding (mapa_estado_civil.pkl) com joblib
🔹 3. Interface web com Streamlit
Aplicação interativa em app.py com layout em duas colunas
Inputs amigáveis para renda, score, estado civil, tempo de emprego, patrimônio e dívidas
Carregamento do modelo e do mapa de encoding salvos no treinamento
Predição da classe (Bom Pagador / Alto Risco) e exibição das probabilidades
Painel de transparência mostrando o vetor enviado ao modelo
🔹 4. Containerização com Docker
Imagem base python:3.12-slim
Instalação das dependências via requirements.txt
Treinamento executado durante o build (train_model.py), garantindo que o container já suba com o modelo pronto
Exposição da porta 8501 e execução automática do Streamlit
Boas Práticas Aplicadas
Separação clara entre geração de dados, treinamento e aplicação
Persistência conjunta do modelo e do encoder, evitando inconsistências em produção
Reprodutibilidade garantida por random_state e versionamento de dependências
Deploy reprodutível via Docker, pronto para rodar em qualquer ambiente
Resultado
Uma aplicação web funcional, portátil e empacotada em container, capaz de classificar novos clientes em tempo real entre Bom Pagador e Alto Risco de Inadimplência, com exibição das probabilidades associadas à decisão.
