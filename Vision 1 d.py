from flask import Flask, request, jsonify
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import logging

# Imposta il logging per una migliore visibilità su Render
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# === CONFIGURAZIONE E CARICAMENTO MODELLO LLAMA ===
# NOTA: Usiamo un percorso relativo, essenziale per il deploy su servizi come Render.
percorso_cartella_modelli = "./models"
os.makedirs(percorso_cartella_modelli, exist_ok=True)

repo_id = "Qwen/Qwen2-1.5B-Instruct-GGUF"
nome_file_modello = "qwen2-1_5b-instruct-q4_k_m.gguf"
percorso_completo_modello = os.path.join(percorso_cartella_modelli, nome_file_modello)

# Download del modello se non esiste
if not os.path.exists(percorso_completo_modello):
    logging.info(f"Modello non trovato. Inizio il download di '{nome_file_modello}'...")
    hf_hub_download(
        repo_id=repo_id,
        filename=nome_file_modello,
        local_dir=percorso_cartella_modelli,
        local_dir_use_symlinks=False
    )
    logging.info("Download completato.")
else:
    logging.info(f"Modello già presente in '{percorso_completo_modello}'.")

# Caricamento del modello (avviene una sola volta all'avvio dell'app)
logging.info("Caricamento del modello in memoria...")
try:
    llm = Llama(
        model_path=percorso_completo_modello,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=10, # Imposta a 0 se non usi una GPU su Render
        verbose=False
    )
    logging.info("Modello caricato con successo. L'API è pronta.")
except Exception as e:
    logging.error(f"Errore durante il caricamento del modello: {e}")
    llm = None

# === PROMPT DI SISTEMA ===
# Questo è il prompt iniziale che definisce il comportamento del bot
SYSTEM_PROMPT_MESSAGES = [
    {
        "role": "system",
        "content": (
            "Sei un assistente AI utile e cordiale specializzato nell'istruzione. "
            "Rispondi sempre e solo in italiano. "
            "Alle domande su chi sei rispondi sempre: Sono Vision, un'AI creata da Cla!. "
            "Alle domande relative su chi ti ha creato rispondi sempre: Sono stato creato dal team di Cla!"
        )
    }
]


# === NUOVO ENDPOINT API PER FLUTTERFLOW ===
@app.route('/api/chat', methods=['POST'])
def api_chat():
    if llm is None:
        return jsonify({'error': 'Modello non caricato, impossibile processare la richiesta'}), 503

    # 1. Riceve i dati JSON da FlutterFlow
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Il campo "message" è obbligatorio'}), 400

    user_message = data['message']
    # La cronologia viene passata dal client. Se non c'è, è la prima interazione.
    history = data.get('history', [])

    # 2. Prepara i messaggi per il modello
    # Se la cronologia è vuota, inizia con il prompt di sistema
    if not history:
        messages = SYSTEM_PROMPT_MESSAGES.copy()
    else:
        messages = history
    
    # Aggiunge il nuovo messaggio dell'utente
    messages.append({"role": "user", "content": user_message})

    # 3. Ottiene la risposta dal modello (non in streaming)
    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            stream=False # Fondamentale: otteniamo una risposta unica
        )
        bot_reply = response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"Errore nella generazione della risposta del modello: {e}")
        return jsonify({'error': 'Errore interno del server durante la generazione della risposta'}), 500

    # 4. Aggiorna la cronologia con la risposta del bot
    messages.append({"role": "assistant", "content": bot_reply})

    # 5. Restituisce la risposta e la nuova cronologia a FlutterFlow
    return jsonify({
        'reply': bot_reply,
        'new_history': messages
    })


# Mantengo la tua vecchia route per test web, se vuoi, ma non è necessaria per FlutterFlow
@app.route('/')
def index():
    return "<h1>Chatbot API</h1><p>Questa è l'API del chatbot. Usa l'endpoint /api/chat con una richiesta POST.</p>"


if __name__ == '__main__':
    # Render imposterà la variabile PORT automaticamente
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host='0.0.0.0', port=port)





