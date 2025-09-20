# Base image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Aggiorna apt e installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice
COPY . .

# Esponi porta
EXPOSE 5000

# Avvia app
CMD ["python", "app.py"]
