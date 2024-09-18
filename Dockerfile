# Utiliza uma imagem base oficial do Python
FROM python:3.9-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências para o container
COPY requirements.txt requirements.txt

# Instala as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante dos arquivos do projeto para o container
COPY . .

# Define a porta que o container vai expor (se necessário)
# EXPOSE 5000  # Descomente se for uma aplicação web que expõe uma porta

# Comando para rodar o script Python quando o container for iniciado
CMD ["python", "previsai.py"]
