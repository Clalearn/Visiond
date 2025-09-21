from flask import Flask, request, jsonify
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# === CONFIGURAZIONE MODELLO LLAMA ===
# Salva il modello nella cartella dell'app (compatibile con Render e Linux)
percorso_cartella_modelli = os.path.join(os.getcwd(), "chatbot_models")
os.makedirs(percorso_cartella_modelli, exist_ok=True)

repo_id = "Qwen/Qwen2-1.5B-Instruct-GGUF"
nome_file_modello = "qwen2-1_5b-instruct-q4_k_m.gguf"

percorso_completo_modello = os.path.join(percorso_cartella_modelli, nome_file_modello)

# Scarica il modello solo se non esiste
if not os.path.exists(percorso_completo_modello):
    print(f"[INFO] Modello non trovato. Download di '{nome_file_modello}' in '{percorso_cartella_modelli}'...")
    hf_hub_download(
        repo_id=repo_id,
        filename=nome_file_modello,
        local_dir=percorso_cartella_modelli,
        local_dir_use_symlinks=False
    )
    print("[INFO] Download completato.")
else:
    print(f"[INFO] Modello già presente in '{percorso_completo_modello}'.")

print("[INFO] Caricamento del modello in memoria... Potrebbe richiedere un po' di tempo.")
llm = Llama(
    model_path=percorso_completo_modello,
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=10,
    verbose=False
)
print("[INFO] Modello caricato correttamente. Pronto per chattare!")

# === CRONOLOGIA CHAT ===
cronologia_chat = [
    {
        "role": "system",
        "content": (
            "Sei un assistente AI utile e cordiale specializzato nell'istruzione. "
            "Rispondi sempre e solo in italiano. "
            "Alle domande su chi sei rispondi sempre: Sono l'AI di Cla!. "
            "Alle domande relative su chi ti ha creato rispondi sempre: Sono stato creato dal team di Cla!"
        )
    }
]

# === FLASK APP ===
app = Flask(__name__)

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cla! Chatbot</title>
        <style>
            body {
                font-family: sans-serif;
                background-color: #f0f0f0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
            }
            .chat-container {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                overflow: hidden;
                width: 80%;
                max-width: 600px;
                display: flex;
                flex-direction: column;
            }
            .chat-header {
                padding: 15px;
                text-align: center;
                border-bottom: 1px solid #eee;
            }
            .chat-header img {
                max-width: 150px;
            }
            .chat-log {
                padding: 15px;
                flex-grow: 1;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
            }
            .message {
                padding: 8px 12px;
                margin-bottom: 8px;
                border-radius: 15px;
                clear: both;
            }
            .user-message {
                background-color: #e0f7fa;
                align-self: flex-end;
                color: #00838f;
            }
            .bot-message {
                background-color: #f5f5f5;
                color: #333;
                align-self: flex-start;
            }
            .input-area {
                padding: 10px;
                display: flex;
                border-top: 1px solid #eee;
            }
            #user-input {
                flex-grow: 1;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-right: 10px;
            }
            button {
                background-color: #00838f;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #006064;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <img src="/static/logo_prova1.png" alt="Logo Cla!">
            </div>
            <div class="chat-log" id="chat-log">
                <div class="message bot-message">Ciao! Sono Cla, la tua assistente AI. Come posso aiutarti oggi?</div>
            </div>
            <div class="input-area">
                <input type="text" id="user-input" placeholder="Scrivi qui il tuo messaggio..." autofocus>
                <button type="button" onclick="sendMessage()">Invia</button>
            </div>
        </div>

        <script>
            function sendMessage() {
                const userInput = document.getElementById('user-input').value.trim();
                if (!userInput) return;

                const chatLog = document.getElementById('chat-log');
                chatLog.innerHTML += `<div class="message user-message">Utente: ${userInput}</div>`;
                document.getElementById('user-input').value = '';
                chatLog.scrollTop = chatLog.scrollHeight;

                fetch('/get_response', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ 'message': userInput })
                })
                .then(response => response.json())
                .then(data => {
                    chatLog.innerHTML += `<div class="message bot-message">Cla!: ${data.response}</div>`;
                    chatLog.scrollTop = chatLog.scrollHeight;
                });
            }

            document.getElementById('user-input').addEventListener('keypress', function (e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data['message'].strip()
    print(f"[LOG] Messaggio ricevuto: {user_input}", flush=True)

    cronologia_chat.append({"role": "user", "content": user_input})

    try:
        output = llm.create_chat_completion(
            messages=cronologia_chat,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stream=False
        )
        risposta = output['choices'][0]['message']['content']
        cronologia_chat.append({"role": "assistant", "content": risposta})
        print(f"[LOG] Risposta generata: {risposta[:100]}...", flush=True)
    except Exception as e:
        risposta = f"Si è verificato un errore durante l'elaborazione: {str(e)}"
        print(f"[ERRORE] {e}", flush=True)

    return jsonify({'response': risposta})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


