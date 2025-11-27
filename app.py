import os
import io
import time
import json
import uuid
import faiss
import numpy as np
import pickle
from collections import deque
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template_string
from supabase import create_client, Client

# ====================================================================
# CONFIGURAÇÃO E INICIALIZAÇÃO
# ====================================================================

app = Flask(__name__)

# Configurações Globais
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
BUCKET_NAME = "teach-ai-files" # Nome do bucket criado no Supabase

# Configuração de Ambiente (Render ou Local)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") # Use a chave 'service_role'
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Inicialização do Cliente Supabase
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Conectado ao Supabase.")
    except Exception as e:
        print(f"Erro ao conectar Supabase: {e}")
else:
    print("AVISO: Variáveis SUPABASE_URL e SUPABASE_KEY não encontradas.")

# Inicialização da IA
embedder = SentenceTransformer(EMBED_MODEL)

# Estado em Memória (Cache)
state = {
    'index': faiss.IndexFlatL2(EMBED_DIM),
    'id_to_text': {}, 
    'docs_meta': {},
    'history': [],
    'req_timestamps': deque(),
    'api_key': GEMINI_API_KEY or ""
}

# ====================================================================
# FUNÇÕES DE PERSISTÊNCIA (SUPABASE)
# ====================================================================

def download_file_from_supabase(filename):
    """Baixa um arquivo do Storage para a memória."""
    if not supabase: return None
    try:
        response = supabase.storage.from_(BUCKET_NAME).download(filename)
        return response # Retorna bytes
    except Exception as e:
        print(f"Arquivo {filename} não encontrado ou erro ao baixar: {e}")
        return None

def upload_file_to_supabase(filename, data_bytes, content_type='application/octet-stream'):
    """Sobe um arquivo para o Storage (sobrescreve se existir)."""
    if not supabase: return
    try:
        # Opção upsert=True força a sobrescrever
        supabase.storage.from_(BUCKET_NAME).upload(
            path=filename,
            file=data_bytes,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        print(f"Salvo no Supabase: {filename}")
    except Exception as e:
        print(f"Erro ao salvar {filename}: {e}")

def load_state():
    """Restaura o estado do aplicativo do Supabase."""
    print("Carregando estado do Supabase...")
    
    # 1. Carregar Índice FAISS
    index_bytes = download_file_from_supabase('vector_store.index')
    if index_bytes:
        with open('/tmp/temp_index.faiss', 'wb') as f:
            f.write(index_bytes)
        state['index'] = faiss.read_index('/tmp/temp_index.faiss')
    
    # 2. Carregar Textos (JSON)
    text_bytes = download_file_from_supabase('id_to_text.json')
    if text_bytes:
        data = json.loads(text_bytes.decode('utf-8'))
        state['id_to_text'] = {int(k): v for k, v in data.items()}

    # 3. Carregar Metadados (JSON)
    meta_bytes = download_file_from_supabase('docs_meta.json')
    if meta_bytes:
        state['docs_meta'] = json.loads(meta_bytes.decode('utf-8'))

    # 4. Carregar Histórico (JSON)
    hist_bytes = download_file_from_supabase('history.json')
    if hist_bytes:
        state['history'] = json.loads(hist_bytes.decode('utf-8'))
        
    print("Estado carregado.")

def save_state_bg():
    """Salva tudo no Supabase."""
    if not supabase: return

    # 1. Salvar FAISS
    faiss.write_index(state['index'], '/tmp/temp_index.faiss')
    with open('/tmp/temp_index.faiss', 'rb') as f:
        upload_file_to_supabase('vector_store.index', f.read())

    # 2. Salvar Textos
    json_texts = json.dumps(state['id_to_text'])
    upload_file_to_supabase('id_to_text.json', json_texts.encode('utf-8'), 'application/json')

    # 3. Salvar Metadados
    json_meta = json.dumps(state['docs_meta'])
    upload_file_to_supabase('docs_meta.json', json_meta.encode('utf-8'), 'application/json')

    # 4. Salvar Histórico
    json_hist = json.dumps(state['history'])
    upload_file_to_supabase('history.json', json_hist.encode('utf-8'), 'application/json')

def process_pdf_bytes(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = "\n\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        chunks = [full_text[i:i + CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE - CHUNK_OVERLAP)]
        return chunks, len(reader.pages)
    except Exception as e:
        print(f"Erro PDF: {e}")
        return [], 0

def call_gemini_api(prompt):
    if not state['api_key']: return {'error': 'API Key não configurada.'}
    
    now = time.time()
    while state['req_timestamps'] and (now - state['req_timestamps'][0] > 60):
        state['req_timestamps'].popleft()
    if len(state['req_timestamps']) >= 15:
        return {'error': 'Muitas requisições. Aguarde.'}

    try:
        genai.configure(api_key=state['api_key'])
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        state['req_timestamps'].append(time.time())
        return {'text': response.text}
    except Exception as e:
        return {'error': str(e)}

# Carregar ao iniciar
with app.app_context():
    load_state()

# ====================================================================
# ROTAS API
# ====================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    conversation_id = data.get('conversation_id')
    
    results = []
    if state['index'].ntotal > 0:
        q_vec = embedder.encode([question], convert_to_numpy=True).astype('float32')
        dists, idxs = state['index'].search(q_vec, DEFAULT_TOP_K)
        for i, idx in enumerate(idxs[0]):
            if idx != -1 and int(idx) in state['id_to_text']:
                results.append({'text': state['id_to_text'][int(idx)], 'score': float(dists[0][i])})
    
    context_str = "\n---\n".join([r['text'] for r in results]) if results else "Sem contexto."
    style = "Dê a resposta direta." if data.get('give_final') else "Dê apenas uma dica/guia, não a resposta final."
    
    prompt = f"Instrutor: {style}\nMatéria: {data.get('subject')}\nContexto:\n{context_str}\n\nAluno: {question}"
    
    resp = call_gemini_api(prompt)
    if 'error' in resp: return jsonify(resp), 500
    
    new_turn = {
        'question': question, 'answer': resp['text'], 
        'subject': data.get('subject'), 'timestamp': time.time()
    }
    
    target_convo = next((c for c in state['history'] if c['id'] == conversation_id), None)
    if not target_convo:
        conversation_id = str(uuid.uuid4())
        target_convo = {'id': conversation_id, 'turns': []}
        state['history'].append(target_convo)
    
    target_convo['turns'].append(new_turn)
    save_state_bg()
    
    return jsonify({'conversation_id': conversation_id, 'history': target_convo['turns']})

@app.route('/api/documents', methods=['GET', 'POST'])
def docs_handler():
    if request.method == 'POST':
        files = request.files.getlist('files')
        count = 0
        for f in files:
            content = f.read()
            chunks, pages = process_pdf_bytes(content)
            if chunks:
                vecs = embedder.encode(chunks, convert_to_numpy=True).astype('float32')
                start = state['index'].ntotal
                state['index'].add(vecs)
                for i, txt in enumerate(chunks): state['id_to_text'][start+i] = txt
                state['docs_meta'][f.filename] = {'pages': pages, 'chunks': len(chunks)}
                count += 1
        
        if count > 0: save_state_bg()
        return jsonify({'message': f'{count} processados.'})
    
    return jsonify({'docs_meta': state['docs_meta']})

@app.route('/api/history', methods=['GET'])
def history_handler():
    summary = []
    for c in reversed(state['history'][-20:]):
        if c['turns']:
            summary.append({'id': c['id'], 'turns': [{'question': c['turns'][0]['question'], 'subject': c['turns'][0]['subject']}]})
    return jsonify({'history': summary})

@app.route('/api/config', methods=['GET', 'POST'])
def config_handler():
    if request.method == 'POST': return jsonify({'msg': 'Salvo localmente'})
    return jsonify({'api_key': state['api_key'], 'subjects': ['Matemática', 'Física', 'Química', 'História', 'Português', 'Inglês', 'Outros']})

@app.route('/api/clear_index', methods=['POST'])
def clear_handler():
    state['index'] = faiss.IndexFlatL2(EMBED_DIM)
    state['id_to_text'] = {}
    state['docs_meta'] = {}
    save_state_bg()
    return jsonify({'message': 'Índice limpo.'})

# ====================================================================
# TEMPLATE HTML (COLE SEU HTML ANTIGO AQUI EMBAIXO)
# ====================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Teach AI</title>
    
    <!-- Tailwind CSS for rapid styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Google Fonts: Inter -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Font Awesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

    <style>
        /* Base styles for font and body */
        body {
            font-family: 'Inter', sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            background-color: #f0f2f5; /* A slightly lighter gray */
        }
        /* Hides the scrollbar but keeps functionality */
        .no-scrollbar::-webkit-scrollbar { display: none; }
        .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }

        /* Styles for modals */
        .modal-overlay { transition: opacity 0.3s ease; }
        .modal-container { transition: transform 0.3s ease, opacity 0.3s ease; transform: scale(0.95); opacity: 0; }
        .modal-container.active { transform: scale(1); opacity: 1; }

        /* Responsive Styles */
        #app-wrapper {
            width: 100%;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #app-container {
            width: 100%;
            height: 100%;
            max-height: 100vh;
            border-radius: 0;
            box-shadow: none;
            display: flex;
            flex-direction: column;
            background-color: #ffffff;
        }
        
        #main-content {
            transition: margin-left 0.3s ease-in-out;
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
        }

        /* Desktop Styles */
        @media (min-width: 768px) {
            #app-container {
                width: 100%;
                max-width: 1100px; 
                height: 90vh;
                max-height: 850px;
                border-radius: 1.5rem; /* rounded-3xl */
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25); /* shadow-2xl */
                flex-direction: row;
            }

            #sidebar {
                position: static;
                transform: translateX(0);
                transition: width 0.3s ease-in-out, padding 0.3s ease-in-out;
                width: 288px; /* w-72 */
                padding: 1.5rem; /* p-6 */
                overflow: hidden;
            }
            
            #main-content-wrapper {
                flex-grow: 1;
                overflow: hidden;
                border-radius: 0 1.5rem 1.5rem 0;
            }
        }
    </style>
</head>
<body class="overflow-hidden">

    <div id="app-wrapper">
        <!-- Main Application Container -->
        <div id="app-container" class="overflow-hidden relative">

            <!-- ====================================================== -->
            <!-- SIDEBAR                                                -->
            <!-- ====================================================== -->
            <div id="sidebar" class="absolute md:relative top-0 left-0 h-full w-72 bg-gray-800 text-white transform -translate-x-full md:translate-x-0 transition-transform duration-300 ease-in-out z-30 p-6 flex flex-col">
                <div id="sidebar-content" class="flex flex-col h-full">
                    <div class="flex justify-between items-center mb-6">
                        <h2 class="text-2xl font-bold text-white">Teach AI</h2>
                        <button onclick="startNewConversation()" title="Nova Conversa" class="p-2 rounded-full hover:bg-gray-700">
                            <i class="fas fa-plus"></i>
                        </button>
                    </div>
                    
                    <h3 class="text-sm font-semibold text-gray-400 uppercase mb-2">Histórico</h3>
                    <div id="history-list" class="flex-grow overflow-y-auto space-y-2 pr-2">
                        <!-- History items will be injected by JS -->
                        <p class="text-gray-500 text-sm">Nenhuma conversa ainda.</p>
                    </div>
                    
                    <div class="border-t border-gray-600 pt-4 mt-4 space-y-2">
                         <button onclick="showScreen('dev-screen')" class="w-full text-left flex items-center gap-3 p-2 rounded-lg hover:bg-gray-700 transition-colors">
                            <i class="fas fa-tools w-5 text-center"></i>
                            <span>Desenvolvedor</span>
                        </button>
                        <button onclick="showScreen('start-screen')" class="w-full text-left flex items-center gap-3 p-2 rounded-lg hover:bg-gray-700 transition-colors">
                           <i class="fas fa-sign-out-alt w-5 text-center"></i>
                            <span>Sair</span>
                        </button>
                    </div>
                </div>
            </div>

            <div id="main-content-wrapper" class="relative flex flex-col w-full h-full bg-gray-50">
                <!-- Sidebar Overlay (mobile only) -->
                <div id="sidebar-overlay" class="hidden md:hidden absolute inset-0 bg-black bg-opacity-50 z-20" onclick="toggleSidebar()"></div>
                
                <!-- Main Content (Screens) -->
                <div id="main-content" class="flex-grow">
                    <!-- ====================================================== -->
                    <!-- SCREEN 1: START SCREEN                                 -->
                    <!-- ====================================================== -->
                    <div id="start-screen" class="flex flex-col h-full p-8 text-center bg-white">
                        <div class="flex-grow flex flex-col items-center justify-center">
                            <div class="w-48 h-48 bg-indigo-100 rounded-full flex items-center justify-center mb-8">
                                <i class="fas fa-robot text-7xl text-indigo-500"></i>
                            </div>
                            <h1 class="text-4xl font-bold text-gray-800">Bem-vindo ao Teach AI</h1>
                            <p class="text-gray-500 mt-2 text-lg">Seu assistente de estudos pessoal.</p>
                        </div>
                        <button onclick="startNewConversation()" class="w-full bg-indigo-600 text-white py-4 rounded-xl font-semibold text-lg hover:bg-indigo-700 transition-colors flex items-center justify-center gap-2">
                            Começar
                            <i class="fas fa-arrow-right"></i>
                        </button>
                    </div>

                    <!-- ====================================================== -->
                    <!-- SCREEN 2: INITIAL CHAT SCREEN (EMPTY)                  -->
                    <!-- ====================================================== -->
                    <div id="chat-initial-screen" class="hidden flex-col h-full">
                        <header class="flex items-center p-4 border-b border-gray-200 flex-shrink-0">
                            <button onclick="toggleSidebar()" class="p-2 rounded-full hover:bg-gray-100 md:hidden">
                                <i class="fas fa-bars text-gray-700"></i>
                            </button>
                            <div class="text-center flex-grow">
                                <h2 class="font-semibold text-gray-800">Nova Conversa</h2>
                                <p id="subject-header-text" class="text-xs text-green-600 font-semibold">Nenhuma matéria selecionada</p>
                            </div>
                            <div class="w-8"></div> <!-- Spacer -->
                        </header>
                        <main class="flex-grow flex flex-col items-center justify-center p-4 text-center">
                            <i class="fas fa-comments text-5xl text-gray-300 mb-4"></i>
                            <h3 class="text-xl font-semibold text-gray-700">Como posso ajudar?</h3>
                            <p class="text-gray-400">Selecione uma matéria e faça sua primeira pergunta abaixo.</p>
                        </main>
                        <footer class="p-4 border-t border-gray-200 flex-shrink-0 bg-white">
                            <div class="flex justify-center gap-4 mb-3">
                                <button onclick="showModal('subject-modal')" class="text-sm text-indigo-600 hover:underline font-medium">
                                    <i class="fas fa-book mr-1"></i><span id="subject-btn-text">Escolher Matéria</span>
                                </button>
                                <button id="question-type-btn-initial" onclick="showModal('question-type-modal')" class="text-sm text-indigo-600 hover:underline font-medium">
                                    <i class="fas fa-tag mr-1"></i><span>Tipo de Pergunta</span>
                                </button>
                                <button id="answer-style-btn-initial" onclick="showModal('answer-style-modal')" class="text-sm text-indigo-600 hover:underline font-medium">
                                    <i class="fas fa-lightbulb mr-1"></i><span>Estilo da Resposta</span>
                                </button>
                            </div>
                            <div class="flex items-center gap-2">
                                <textarea id="initial-chat-input" placeholder="Digite sua dúvida aqui..." class="w-full p-3 bg-gray-100 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none" rows="1"></textarea>
                                <button onclick="startNewChat()" class="bg-indigo-600 text-white p-3 rounded-xl hover:bg-indigo-700 transition-colors flex items-center justify-center">
                                    <i class="fas fa-paper-plane"></i>
                                </button>
                            </div>
                        </footer>
                    </div>

                    <!-- ====================================================== -->
                    <!-- SCREEN 3: ACTIVE CHAT SCREEN                           -->
                    <!-- ====================================================== -->
                    <div id="chat-conversa-screen" class="hidden flex-col h-full">
                        <header class="flex items-center p-4 border-b border-gray-200 flex-shrink-0">
                             <button onclick="toggleSidebar()" class="p-2 rounded-full hover:bg-gray-100 md:hidden">
                                <i class="fas fa-bars text-gray-700"></i>
                            </button>
                            <div class="text-center flex-grow">
                                <h2 class="font-semibold text-gray-800">Conversa Ativa</h2>
                                <p id="subject-header-text-chat" class="text-xs text-green-600 font-semibold">N/A</p>
                            </div>
                            <div class="w-8"></div> <!-- Spacer -->
                        </header>
                        <main id="chat-body" class="flex-grow p-4 overflow-y-auto no-scrollbar space-y-4">
                            <!-- Messages will be added here via JS -->
                        </main>
                        <footer class="p-4 border-t border-gray-200 flex-shrink-0 bg-white">
                             <div class="flex justify-center gap-4 mb-3">
                                <button onclick="showModal('subject-modal')" class="text-sm text-indigo-600 hover:underline font-medium">
                                    <i class="fas fa-book mr-1"></i><span id="subject-btn-text-convo"></span>
                                </button>
                                <button id="question-type-btn-conversa" onclick="showModal('question-type-modal')" class="text-sm text-indigo-600 hover:underline font-medium">
                                    <i class="fas fa-tag mr-1"></i><span></span>
                                </button>
                                <button id="answer-style-btn-conversa" onclick="showModal('answer-style-modal')" class="text-sm text-indigo-600 hover:underline font-medium">
                                    <i class="fas fa-lightbulb mr-1"></i><span></span>
                                </button>
                            </div>
                            <div class="flex items-center gap-2">
                                <textarea id="chat-conversa-input" placeholder="Faça outra pergunta..." class="w-full p-3 bg-gray-100 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none" rows="1"></textarea>
                                <button onclick="sendMessage()" class="bg-indigo-600 text-white p-3 rounded-xl hover:bg-indigo-700 transition-colors flex items-center justify-center">
                                    <i class="fas fa-paper-plane"></i>
                                </button>
                            </div>
                        </footer>
                    </div>

                    <!-- ====================================================== -->
                    <!-- SCREEN 4: DEVELOPER SCREEN                             -->
                    <!-- ====================================================== -->
                    <div id="dev-screen" class="hidden flex-col h-full">
                        <header class="flex items-center p-4 border-b border-gray-200 flex-shrink-0">
                            <button onclick="goBack()" class="p-2 rounded-full hover:bg-gray-100">
                                <i class="fas fa-arrow-left text-gray-700"></i>
                            </button>
                            <div class="text-center flex-grow">
                                <h2 class="font-semibold text-gray-800">Painel do Desenvolvedor</h2>
                            </div>
                            <div class="w-8"></div>
                        </header>
                        <main class="flex-grow p-6 space-y-6 overflow-y-auto">
                            <!-- API Key Configuration -->
                            <div class="bg-white p-4 rounded-lg shadow-sm border">
                                <h3 class="font-semibold text-lg mb-2">🔑 Configuração da API do Google Gemini</h3>
                                <p class="text-sm text-gray-600 mb-3">Sua chave de API é salva localmente no seu navegador e usada para se comunicar com o Gemini.</p>
                                <input id="api-key-input" type="password" placeholder="Cole sua chave de API aqui" class="w-full p-2 border rounded-md">
                                <button onclick="saveApiKey()" class="mt-2 w-full bg-indigo-600 text-white py-2 rounded-md hover:bg-indigo-700">Salvar Chave</button>
                            </div>

                            <!-- Document Management -->
                            <div class="bg-white p-4 rounded-lg shadow-sm border">
                                <h3 class="font-semibold text-lg mb-2">📚 Base de Conhecimento (PDFs)</h3>
                                <div class="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                                    <input type="file" id="pdf-upload" multiple accept=".pdf" class="hidden">
                                    <label for="pdf-upload" class="cursor-pointer">
                                        <i class="fas fa-cloud-upload-alt text-4xl text-gray-400 mb-2"></i>
                                        <p class="text-gray-600">Arraste e solte arquivos PDF aqui ou clique para selecionar.</p>
                                    </label>
                                </div>
                                <button onclick="uploadDocuments()" class="mt-3 w-full bg-green-600 text-white py-2 rounded-md hover:bg-green-700">Adicionar Documentos</button>
                            </div>
                            
                             <!-- Current Documents -->
                            <div class="bg-white p-4 rounded-lg shadow-sm border">
                                <h3 class="font-semibold text-lg mb-2">📄 Documentos Atuais</h3>
                                <div id="documents-list" class="space-y-2">
                                    <!-- Document list will be populated here -->
                                </div>
                            </div>

                            <!-- Maintenance -->
                            <div class="bg-white p-4 rounded-lg shadow-sm border">
                                <h3 class="font-semibold text-lg mb-2">🛠️ Manutenção</h3>
                                <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                                    <button onclick="reindex()" class="bg-yellow-500 text-white py-2 rounded-md hover:bg-yellow-600">Reindexar Docs</button>
                                    <button onclick="clearIndex()" class="bg-red-600 text-white py-2 rounded-md hover:bg-red-700">Limpar Índice</button>
                                    <button onclick="exportHistory()" class="bg-blue-500 text-white py-2 rounded-md hover:bg-blue-600 col-span-1 md:col-span-2">Exportar Histórico</button>
                                </div>
                            </div>
                        </main>
                    </div>
                </div>
                
                <!-- Modal Overlay -->
                <div id="modal-overlay" class="hidden absolute inset-0 bg-black bg-opacity-50 z-40 modal-overlay" onclick="hideAllModals()"></div>

                <!-- ====================================================== -->
                <!-- MODALS                                                 -->
                <!-- ====================================================== -->
                <div id="question-type-modal" class="hidden absolute inset-0 z-50 flex items-center justify-center p-4">
                    <div class="bg-white rounded-xl shadow-lg p-6 w-full max-w-xs modal-container">
                        <h3 class="text-lg font-semibold text-center mb-4">Tipo de pergunta</h3>
                        <div class="space-y-2" id="question-type-options">
                           <!-- Options populated by JS -->
                        </div>
                    </div>
                </div>

                <div id="answer-style-modal" class="hidden absolute inset-0 z-50 flex items-center justify-center p-4">
                    <div class="bg-white rounded-xl shadow-lg p-6 w-full max-w-xs modal-container">
                        <h3 class="text-lg font-semibold text-center mb-4">Estilo da resposta</h3>
                        <div class="space-y-2">
                            <button onclick="selectAnswerStyle(false)" class="w-full text-left p-3 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors">
                                <strong class="block">Dica / Explicação</strong>
                                <span class="text-xs text-gray-500">Me guie para a resposta.</span>
                            </button>
                            <button onclick="selectAnswerStyle(true)" class="w-full text-left p-3 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors">
                                <strong class="block">Resposta Direta</strong>
                                <span class="text-xs text-gray-500">Me dê a solução completa.</span>
                            </button>
                        </div>
                    </div>
                </div>

                <div id="subject-modal" class="hidden absolute inset-0 z-50 flex items-center justify-center p-4">
                    <div class="bg-white rounded-xl shadow-lg p-6 w-full max-w-xs modal-container">
                        <h3 class="text-lg font-semibold text-center mb-4">Escolha a matéria</h3>
                        <div class="space-y-2" id="subject-options">
                            <!-- Options populated by JS -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- JAVASCRIPT LOGIC -->
    <script>
        // --- STATE AND ELEMENTS ---
        const state = {
            currentScreen: 'start-screen',
            previousScreen: 'start-screen',
            isDesktop: window.innerWidth >= 768,
            isSidebarOpen: window.innerWidth >= 768,
            apiKey: localStorage.getItem('geminiApiKey') || '',
            activeConversationId: null,
            conversations: {},
            // Default selections
            selectedSubject: 'Outros',
            selectedQType: 'Dúvida sobre o livro',
            selectedAnswerStyle: false, // false for 'hint', true for 'full'
            // Constants from backend
            subjects: [],
            questionTypes: ['Dúvida sobre o livro', 'Pergunta da aula', 'Conceito específico', 'Curiosidade', 'Outro'],
        };

        const elements = {
            appContainer: document.getElementById('app-container'),
            sidebar: document.getElementById('sidebar'),
            sidebarOverlay: document.getElementById('sidebar-overlay'),
            mainContent: document.getElementById('main-content'),
            initialChatInput: document.getElementById('initial-chat-input'),
            chatConversaInput: document.getElementById('chat-conversa-input'),
            chatBody: document.getElementById('chat-body'),
            apiKeyInput: document.getElementById('api-key-input'),
            documentsList: document.getElementById('documents-list'),
            historyList: document.getElementById('history-list'),
            pdfUpload: document.getElementById('pdf-upload'),
            subjectOptions: document.getElementById('subject-options'),
            questionTypeOptions: document.getElementById('question-type-options'),
        };

        const allScreens = ['start-screen', 'chat-initial-screen', 'chat-conversa-screen', 'dev-screen'];
        const allModals = ['question-type-modal', 'subject-modal', 'answer-style-modal'];

        // --- API HELPERS ---
        async function apiCall(endpoint, method = 'GET', body = null) {
            const options = {
                method,
                headers: { 'Content-Type': 'application/json' },
            };
            if (body) {
                options.body = body instanceof FormData ? body : JSON.stringify(body);
                if (body instanceof FormData) {
                    delete options.headers['Content-Type']; // Let browser set it for FormData
                }
            }
            try {
                const response = await fetch(`/api${endpoint}`, options);
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
                }
                return await response.json();
            } catch (error) {
                console.error(`API call to ${endpoint} failed:`, error);
                alert(`Erro: ${error.message}`);
                return null;
            }
        }

        // --- UI & SCREEN MANAGEMENT ---

        function showScreen(screenId) {
            if (state.currentScreen === screenId) return;

            allScreens.forEach(id => document.getElementById(id)?.classList.add('hidden'));
            
            const targetScreen = document.getElementById(screenId);
            if (targetScreen) {
                targetScreen.classList.remove('hidden');
                targetScreen.classList.add('flex');
                state.previousScreen = state.currentScreen;
                state.currentScreen = screenId;
            }
            if (!state.isDesktop) {
                toggleSidebar(false);
            }
        }

        function goBack() {
            showScreen(state.previousScreen || 'chat-initial-screen');
        }

        function toggleSidebar(force) {
            state.isSidebarOpen = typeof force === 'boolean' ? force : !state.isSidebarOpen;

            if (state.isDesktop) {
                // On desktop, we don't hide, we just control layout if needed
                // This version keeps it static on desktop
            } else {
                elements.sidebar.classList.toggle('-translate-x-full', !state.isSidebarOpen);
                elements.sidebarOverlay.classList.toggle('hidden', !state.isSidebarOpen);
            }
        }
        
        function showModal(modalId) {
            document.getElementById('modal-overlay').classList.remove('hidden');
            const modal = document.getElementById(modalId);
            modal.classList.remove('hidden');
            modal.classList.add('flex');
            setTimeout(() => modal.querySelector('.modal-container')?.classList.add('active'), 10);
        }

        function hideAllModals() {
            document.getElementById('modal-overlay').classList.add('hidden');
            allModals.forEach(id => {
                const modal = document.getElementById(id);
                if (modal) {
                    modal.querySelector('.modal-container')?.classList.remove('active');
                    setTimeout(() => {
                        modal.classList.add('hidden');
                        modal.classList.remove('flex');
                    }, 300);
                }
            });
        }

        function handleResize() {
            const wasDesktop = state.isDesktop;
            state.isDesktop = window.innerWidth >= 768;
            if (wasDesktop !== state.isDesktop) {
                toggleSidebar(state.isDesktop); // Open on desktop, close on mobile
            }
        }

        // --- CHAT LOGIC ---

        function startNewConversation() {
            state.activeConversationId = null;
            showScreen('chat-initial-screen');
        }

        async function startNewChat() {
            const question = elements.initialChatInput.value.trim();
            if (!question) return;
            if (!state.apiKey) {
                alert("Por favor, configure sua API Key na aba de Desenvolvedor.");
                return;
            }

            elements.initialChatInput.value = '';
            showScreen('chat-conversa-screen');
            
            appendMessage(question, 'user');
            appendMessage('', 'ai', true); // Show loading indicator

            const response = await apiCall('/chat', 'POST', {
                question,
                subject: state.selectedSubject,
                qtype: state.selectedQType,
                give_final: state.selectedAnswerStyle,
                conversation_id: null // Starts a new conversation
            });

            if (response && response.conversation_id) {
                state.activeConversationId = response.conversation_id;
                state.conversations[response.conversation_id] = response.history;
                updateChatUI();
                await loadHistory(); // Refresh history list
            } else {
                // Handle error
                removeLoadingMessage();
                appendMessage("Desculpe, ocorreu um erro ao processar sua pergunta.", 'ai');
            }
        }

        async function sendMessage() {
            const question = elements.chatConversaInput.value.trim();
            if (!question) return;

            elements.chatConversaInput.value = '';
            appendMessage(question, 'user');
            appendMessage('', 'ai', true); // Loading

            const response = await apiCall('/chat', 'POST', {
                question,
                subject: state.selectedSubject,
                qtype: state.selectedQType,
                give_final: state.selectedAnswerStyle,
                conversation_id: state.activeConversationId
            });

            if (response) {
                state.conversations[state.activeConversationId] = response.history;
                updateChatUI();
            } else {
                removeLoadingMessage();
                appendMessage("Desculpe, ocorreu um erro.", 'ai');
            }
        }

        function appendMessage(text, sender, isLoading = false) {
            const messageDiv = document.createElement('div');
            const messageBubble = document.createElement('div');
            const icon = document.createElement('i');

            messageDiv.className = `flex items-start gap-2.5 ${sender === 'user' ? 'justify-end' : 'justify-start'}`;
            
            if (sender === 'user') {
                icon.className = 'fas fa-user-circle text-2xl text-gray-400';
            } else {
                icon.className = 'fas fa-robot text-2xl text-indigo-500';
            }

            messageBubble.className = `p-3 rounded-xl max-w-xl ${sender === 'user' ? 'bg-indigo-600 text-white' : 'bg-gray-200 text-gray-800'}`;
            
            if (isLoading) {
                messageBubble.id = 'loading-bubble';
                messageBubble.innerHTML = '<div class="flex items-center justify-center gap-2"><span>Pensando...</span><div class="w-2 h-2 bg-gray-500 rounded-full animate-pulse"></div><div class="w-2 h-2 bg-gray-500 rounded-full animate-pulse" style="animation-delay: 0.2s;"></div><div class="w-2 h-2 bg-gray-500 rounded-full animate-pulse" style="animation-delay: 0.4s;"></div></div>';
            } else {
                // Basic markdown to HTML conversion
                let html = text.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
                html = html.replace(/\\n/g, '<br>');
                messageBubble.innerHTML = html;
            }
            
            if (sender === 'user') {
                messageDiv.appendChild(messageBubble);
                messageDiv.appendChild(icon);
            } else {
                messageDiv.appendChild(icon);
                messageDiv.appendChild(messageBubble);
            }

            elements.chatBody.appendChild(messageDiv);
            elements.chatBody.scrollTop = elements.chatBody.scrollHeight;
        }

        function removeLoadingMessage() {
            document.getElementById('loading-bubble')?.parentElement.remove();
        }

        function updateChatUI() {
            elements.chatBody.innerHTML = '';
            const conversation = state.conversations[state.activeConversationId];
            if (!conversation) return;

            conversation.forEach(turn => {
                appendMessage(turn.question, 'user');
                appendMessage(turn.answer, 'ai');
            });

            // Update header and footer controls
            const lastTurn = conversation[conversation.length - 1];
            selectSubject(lastTurn.subject, true);
            selectQuestionType(lastTurn.qtype, true);
            selectAnswerStyle(lastTurn.final_requested, true);

            elements.chatBody.scrollTop = elements.chatBody.scrollHeight;
        }
        
        function loadConversation(id) {
            state.activeConversationId = id;
            updateChatUI();
            showScreen('chat-conversa-screen');
        }

        // --- DEVELOPER & DATA MANAGEMENT ---

        function saveApiKey() {
            const key = elements.apiKeyInput.value.trim();
            if (key) {
                state.apiKey = key;
                localStorage.setItem('geminiApiKey', key);
                apiCall('/config', 'POST', { api_key: key });
                alert('API Key salva com sucesso!');
            }
        }

        async function uploadDocuments() {
            const files = elements.pdfUpload.files;
            if (files.length === 0) {
                alert('Por favor, selecione arquivos PDF para enviar.');
                return;
            }
            const formData = new FormData();
            for (const file of files) {
                formData.append('files', file);
            }
            
            alert('Enviando documentos... Isso pode levar um momento.');
            const result = await apiCall('/documents', 'POST', formData);
            if (result) {
                alert(`${result.added_count} documento(s) adicionado(s) com sucesso.`);
                loadDocuments();
            }
        }

        async function loadDocuments() {
            const data = await apiCall('/documents');
            elements.documentsList.innerHTML = '';
            if (data && Object.keys(data.docs_meta).length > 0) {
                for (const [name, meta] of Object.entries(data.docs_meta)) {
                    const div = document.createElement('div');
                    div.className = 'flex justify-between items-center p-2 border rounded-md';
                    div.innerHTML = `
                        <div>
                            <p class="font-medium">${name}</p>
                            <p class="text-xs text-gray-500">${meta.pages} páginas | ${meta.n_chunks} chunks</p>
                        </div>
                        <button onclick="deleteDocument('${name}')" class="text-red-500 hover:text-red-700"><i class="fas fa-trash"></i></button>
                    `;
                    elements.documentsList.appendChild(div);
                }
            } else {
                elements.documentsList.innerHTML = '<p class="text-gray-500">Nenhum documento carregado.</p>';
            }
        }

        async function deleteDocument(name) {
            if (confirm(`Tem certeza que deseja remover o documento "${name}"?`)) {
                const result = await apiCall(`/documents/${name}`, 'DELETE');
                if (result) {
                    alert(result.message);
                    loadDocuments();
                }
            }
        }
        
        async function reindex() {
             if (confirm(`Isso irá reprocessar todos os documentos na pasta 'docs'. Deseja continuar?`)) {
                alert('Reindexando... Por favor, aguarde.');
                const result = await apiCall('/reindex', 'POST');
                if(result) alert(result.message);
             }
        }
        
        async function clearIndex() {
            if (confirm(`CUIDADO: Isso apagará todo o índice de conhecimento. Deseja continuar?`)) {
                const result = await apiCall('/clear_index', 'POST');
                if(result) {
                    alert(result.message);
                    loadDocuments();
                }
            }
        }

        async function loadHistory() {
            const data = await apiCall('/history');
            elements.historyList.innerHTML = '';
            if (data && data.history.length > 0) {
                const reversedHistory = [...data.history].reverse();
                state.conversations = {};
                reversedHistory.forEach(convo => {
                    state.conversations[convo.id] = convo.turns;
                    const div = document.createElement('div');
                    const firstQuestion = convo.turns[0]?.question || 'Conversa sem título';
                    const subject = convo.turns[0]?.subject || 'N/A';
                    div.className = 'p-2 rounded-lg hover:bg-gray-700 cursor-pointer';
                    div.innerHTML = `
                        <p class="font-semibold text-white truncate">${firstQuestion}</p>
                        <p class="text-xs text-gray-400">${subject}</p>
                    `;
                    div.onclick = () => loadConversation(convo.id);
                    elements.historyList.appendChild(div);
                });
            } else {
                elements.historyList.innerHTML = '<p class="text-gray-500 text-sm">Nenhuma conversa ainda.</p>';
            }
        }
        
        async function exportHistory() {
            const data = await apiCall('/history');
            if(data && data.history.length > 0) {
                const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data.history, null, 2));
                const downloadAnchorNode = document.createElement('a');
                downloadAnchorNode.setAttribute("href", dataStr);
                downloadAnchorNode.setAttribute("download", "teachai_history.json");
                document.body.appendChild(downloadAnchorNode);
                downloadAnchorNode.click();
                downloadAnchorNode.remove();
            } else {
                alert("Nenhum histórico para exportar.");
            }
        }

        // --- UI SELECTION HANDLERS ---
        function selectQuestionType(type, quiet = false) {
            state.selectedQType = type;
            document.getElementById('question-type-btn-initial').querySelector('span').textContent = type;
            document.getElementById('question-type-btn-conversa').querySelector('span').textContent = type;
            if (!quiet) hideAllModals();
        }

        function selectAnswerStyle(style, quiet = false) {
            state.selectedAnswerStyle = style;
            const text = style ? 'Resposta Direta' : 'Dica / Explicação';
            document.getElementById('answer-style-btn-initial').querySelector('span').textContent = text;
            document.getElementById('answer-style-btn-conversa').querySelector('span').textContent = text;
            if (!quiet) hideAllModals();
        }

        function selectSubject(subject, quiet = false) {
            state.selectedSubject = subject;
            document.getElementById('subject-header-text').textContent = `Matéria: ${subject}`;
            document.getElementById('subject-header-text-chat').textContent = `Matéria: ${subject}`;
            document.getElementById('subject-btn-text').textContent = subject;
            document.getElementById('subject-btn-text-convo').textContent = subject;
            if (!quiet) hideAllModals();
        }

        // --- INITIALIZATION ---
        async function initializeApp() {
            // Load config from backend
            const config = await apiCall('/config');
            if (config) {
                state.subjects = config.subjects;
                if(config.api_key) {
                    state.apiKey = config.api_key;
                    elements.apiKeyInput.value = config.api_key;
                }
            }
            
            // Populate modals
            state.subjects.forEach(s => {
                const btn = document.createElement('button');
                btn.className = "w-full text-center p-3 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors";
                btn.textContent = s;
                btn.onclick = () => selectSubject(s);
                elements.subjectOptions.appendChild(btn);
            });
            state.questionTypes.forEach(q => {
                const btn = document.createElement('button');
                btn.className = "w-full text-center p-3 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors";
                btn.textContent = q;
                btn.onclick = () => selectQuestionType(q);
                elements.questionTypeOptions.appendChild(btn);
            });

            // Set default selections in UI
            selectSubject(state.selectedSubject, true);
            selectQuestionType(state.selectedQType, true);
            selectAnswerStyle(state.selectedAnswerStyle, true);

            // Load dynamic data
            loadDocuments();
            loadHistory();

            // Set up listeners
            window.addEventListener('resize', handleResize);
            [elements.initialChatInput, elements.chatConversaInput].forEach(input => {
                input.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        if (state.currentScreen === 'chat-initial-screen') startNewChat();
                        else if (state.currentScreen === 'chat-conversa-screen') sendMessage();
                    }
                });
            });

            // Initial setup
            showScreen('start-screen');
            handleResize();
        }

        document.addEventListener('DOMContentLoaded', initializeApp);
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)