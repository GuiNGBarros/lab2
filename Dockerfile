#Imagem base
FROM python:3.12-slim

LABEL maintainer="guibarros"

#Pasta de trabalho
WORKDIR /app

#Copiar arquivos da aplicação
COPY . . 

#Instalar dependencias
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN python train_model.py

#Expor porta padrão do streamlit
EXPOSE 8501

#Comando para executar o Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]