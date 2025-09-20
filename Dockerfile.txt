# Usa Python 3.11 slim come base
FROM python:3.11-slim

# Imposta la cartella di lavoro dentro il container
WORKDIR /app

# Copia requirements.txt e installa le dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutti i file del progetto nel container
COPY . .

# Esponi la porta 5000 (usata da Flask)
EXPOSE 5000

# Comando per avviare l'app
CMD ["python", "app.py"]