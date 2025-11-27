<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Teach AI - Assistente Inteligente</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
    <script>pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';</script>
    <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/dist/umd/supabase.min.js"></script>
    <script type="importmap">
      {
        "imports": {
          "@google/generative-ai": "https://esm.run/@google/generative-ai"
        }
      }
    </script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #f3f4f6; }
        .no-scrollbar::-webkit-scrollbar { display: none; }
        .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
        .typing-dot { animation: typing 1.4s infinite ease-in-out both; }
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes typing { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
    </style>
</head>
<body class="h-screen overflow-hidden flex justify-center items-center">

    <div class="w-full h-full md:max-w-6xl md:h-[90vh] bg-white md:rounded-2xl shadow-2xl flex overflow-hidden relative">
        
        <div id="sidebar" class="hidden md:flex flex-col w-72 bg-slate-900 text-white p-6 transition-all duration-300 absolute md:relative z-20 h-full">
            <div class="flex justify-between items-center mb-8">
                <h1 class="text-2xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">Teach AI</h1>
                <button onclick="resetChat()" class="p-2 hover:bg-slate-800 rounded-full transition"><i class="fas fa-plus"></i></button>
            </div>
            
            <div class="flex-grow overflow-y-auto no-scrollbar space-y-2" id="history-container">
                 <div class="p-3 rounded-lg bg-slate-800/50 cursor-pointer hover:bg-slate-800 transition">
                    <p class="text-sm font-medium text-slate-300">Nova conversa...</p>
                </div>
            </div>

            <div class="mt-4 pt-4 border-t border-slate-700 space-y-2">
                <button onclick="toggleScreen('dev-screen')" class="w-full text-left p-2 hover:bg-slate-800 rounded flex items-center gap-3 text-sm text-slate-300">
                    <i class="fas fa-database"></i> Base de Conhecimento
                </button>
                <button onclick="toggleScreen('api-screen')" class="w-full text-left p-2 hover:bg-slate-800 rounded flex items-center gap-3 text-sm text-slate-300">
                    <i class="fas fa-key"></i> Configurar API Key
                </button>
            </div>
        </div>

        <div class="flex-grow flex flex-col relative w-full">
            <div class="md:hidden flex items-center justify-between p-4 border-b bg-white z-10">
                <button onclick="document.getElementById('sidebar').classList.toggle('hidden')" class="text-slate-600"><i class="fas fa-bars text-xl"></i></button>
                <span class="font-bold text-slate-800">Teach AI</span>
                <div class="w-6"></div>
            </div>

            <div id="api-screen" class="hidden absolute inset-0 bg-white z-30 flex-col p-8 justify-center items-center">
                <div class="max-w-md w-full">
                    <h2 class="text-2xl font-bold mb-4 text-center">Configuração Inicial</h2>
                    <p class="text-gray-600 mb-4 text-center">Insira sua chave do Google Gemini para começar.</p>
                    <input type="password" id="api-key-input" placeholder="Cole sua chave AIza..." class="w-full p-3 border rounded mb-4">
                    <button onclick="saveApiKey()" class="bg-indigo-600 text-white px-6 py-3 rounded hover:bg-indigo-700 w-full">Salvar e Entrar</button>
                </div>
            </div>

            <div id="dev-screen" class="hidden absolute inset-0 bg-white z-30 flex-col p-8 overflow-y-auto">
                <div class="flex justify-between items-center mb-6">
                    <h2 class="text-2xl font-bold">Gestão de Documentos (Supabase)</h2>
                    <button onclick="toggleScreen('chat-screen')" class="text-gray-500 hover:text-gray-800"><i class="fas fa-times text-xl"></i></button>
                </div>
                
                <div class="border-2 border-dashed border-indigo-200 rounded-xl p-8 text-center bg-indigo-50 hover:bg-indigo-100 transition cursor-pointer relative">
                    <input type="file" id="file-upload" accept=".pdf" class="absolute inset-0 opacity-0 cursor-pointer" onchange="handleFileUpload(this)">
                    <i class="fas fa-cloud-upload-alt text-4xl text-indigo-400 mb-2"></i>
                    <p class="text-indigo-800 font-medium">Clique para adicionar PDF</p>
                    <p class="text-xs text-indigo-500">O arquivo será sincronizado com a nuvem.</p>
                </div>

                <div class="mt-8">
                    <div class="flex justify-between items-center mb-2">
                         <h3 class="font-bold text-gray-700">Documentos na Nuvem</h3>
                         <button onclick="clearKnowledgeBase()" class="text-xs text-red-500 hover:underline">Limpar Tudo</button>
                    </div>
                   
                    <div id="docs-list" class="space-y-2">
                        <p class="text-sm text-gray-400 italic" id="loading-docs-msg">Verificando Supabase...</p>
                    </div>
                </div>
            </div>

            <div id="chat-screen" class="flex flex-col h-full">
                <div class="p-4 border-b flex justify-between items-center bg-white shadow-sm">
                    <div>
                        <h2 class="font-semibold text-slate-800">Assistente de Estudos</h2>
                        <div class="flex items-center gap-1">
                            <span id="status-dot" class="w-2 h-2 rounded-full bg-yellow-500"></span>
                            <span id="status-text" class="text-xs text-slate-500">Conectando...</span>
                        </div>
                    </div>
                    <div class="flex gap-2">
                        <select id="subject-select" class="text-sm border rounded p-1 text-slate-600 bg-slate-50">
                            <option>Geral</option>
                            <option>Matemática</option>
                            <option>História</option>
                            <option>Física</option>
                        </select>
                    </div>
                </div>

                <div id="chat-box" class="flex-grow overflow-y-auto p-4 space-y-4 bg-slate-50">
                    <div class="flex gap-3">
                        <div class="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center flex-shrink-0">
                            <i class="fas fa-robot text-white text-xs"></i>
                        </div>
                        <div class="bg-white p-4 rounded-2xl rounded-tl-none shadow-sm max-w-[85%] border border-slate-100">
                            <p class="text-slate-700">Olá! Eu sou o Teach AI. Seus documentos estão salvos na nuvem. Pode perguntar!</p>
                        </div>
                    </div>
                </div>

                <div class="p-4 bg-white border-t">
                    <div class="flex items-center gap-2 bg-slate-100 p-2 rounded-xl border border-slate-200 focus-within:border-indigo-500 focus-within:ring-1 focus-within:ring-indigo-500 transition">
                        <button onclick="toggleScreen('dev-screen')" class="p-2 text-slate-400 hover:text-indigo-600 transition" title="Adicionar Documento">
                            <i class="fas fa-paperclip"></i>
                        </button>
                        <input type="text" id="user-input" placeholder="Digite sua dúvida..." class="flex-grow bg-transparent border-none focus:ring-0 text-slate-700 placeholder-slate-400" onkeypress="if(event.key === 'Enter') sendMessage()">
                        <button onclick="sendMessage()" class="p-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition shadow-md">
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script type="module">
        import { GoogleGenerativeAI } from "@google/generative-ai";

        // ================= CONFIGURAÇÃO (PREENCHA AQUI) =================
        const SUPABASE_URL = 'SUA_URL_SUPABASE_AQUI'; 
        const SUPABASE_KEY = 'SUA_KEY_ANON_PUBLIC_AQUI';
        const BUCKET_NAME = 'arquivos-teach-ai';
        const DB_FILE_NAME = 'system/knowledge_base.json'; // O arquivo "cérebro"

        let supabase = null;
        if(SUPABASE_URL && SUPABASE_KEY && SUPABASE_URL !== 'SUA_URL_SUPABASE_AQUI') {
             supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_KEY);
        }

        // Estado Global
        let knowledgeBase = {}; // Objeto { "arquivo.pdf": "texto..." }
        let genAI = null;
        let model = null;

        // Inicialização
        window.addEventListener('load', async () => {
            const savedKey = localStorage.getItem('gemini_key');
            if (savedKey) {
                initGemini(savedKey);
                toggleScreen('chat-screen');
            } else {
                toggleScreen('api-screen');
            }

            // Sincronizar com a nuvem ao abrir
            if(supabase) await loadKnowledgeFromCloud();
        });

        // ================= GESTÃO DE DADOS (PERSISTÊNCIA) =================

        async function loadKnowledgeFromCloud() {
            updateStatus('loading', 'Baixando documentos...');
            const listDiv = document.getElementById('docs-list');
            
            try {
                const { data, error } = await supabase.storage.from(BUCKET_NAME).download(DB_FILE_NAME);
                
                if (error) {
                    console.log("Nenhum conhecimento prévio encontrado.");
                    listDiv.innerHTML = '<p class="text-sm text-gray-400 italic">Nenhum documento na nuvem.</p>';
                    updateStatus('online', 'Online (Vazio)');
                    return;
                }

                const text = await data.text();
                knowledgeBase = JSON.parse(text);
                
                // Renderizar lista
                renderDocsList();
                const count = Object.keys(knowledgeBase).length;
                updateStatus('online', `Online | ${count} docs carregados`);

            } catch (e) {
                console.error("Erro ao carregar:", e);
                updateStatus('error', 'Erro de conexão');
            }
        }

        async function saveKnowledgeToCloud() {
            if (!supabase) return;
            // Salva o objeto knowledgeBase inteiro como um arquivo JSON no Storage
            const blob = new Blob([JSON.stringify(knowledgeBase)], {type : 'application/json'});
            await supabase.storage.from(BUCKET_NAME).upload(DB_FILE_NAME, blob, {
                contentType: 'application/json',
                upsert: true
            });
            console.log("Conhecimento salvo na nuvem!");
        }

        function renderDocsList() {
            const listDiv = document.getElementById('docs-list');
            listDiv.innerHTML = '';
            
            const files = Object.keys(knowledgeBase);
            if(files.length === 0) {
                 listDiv.innerHTML = '<p class="text-sm text-gray-400 italic">Nenhum documento carregado.</p>';
                 return;
            }

            files.forEach(filename => {
                const item = document.createElement('div');
                item.className = "p-3 bg-white border border-slate-200 rounded text-sm flex justify-between items-center";
                item.innerHTML = `
                    <div class="flex items-center gap-2">
                        <i class="fas fa-file-pdf text-red-500"></i>
                        <span class="font-medium text-slate-700 truncate max-w-[150px]">${filename}</span>
                    </div>
                    <span class="text-green-500 text-xs"><i class="fas fa-cloud"></i> Salvo</span>
                `;
                listDiv.appendChild(item);
            });
        }

        window.clearKnowledgeBase = async () => {
            if(confirm("Tem certeza? Isso apagará todos os documentos da memória do Teach AI para todos os usuários.")) {
                knowledgeBase = {};
                renderDocsList();
                await saveKnowledgeToCloud();
                alert("Memória limpa.");
            }
        }

        // ================= UPLOAD E EXTRAÇÃO =================

        window.handleFileUpload = async (input) => {
            const file = input.files[0];
            if (!file) return;

            const listDiv = document.getElementById('docs-list');
            const loadingItem = document.createElement('div');
            loadingItem.className = "p-3 bg-indigo-50 text-indigo-700 rounded text-sm flex justify-between animate-pulse";
            loadingItem.innerHTML = `<span>Lendo ${file.name}...</span><i class="fas fa-spinner fa-spin"></i>`;
            listDiv.prepend(loadingItem);

            try {
                // 1. Extrair Texto
                const arrayBuffer = await file.arrayBuffer();
                const text = await extractTextFromPDF(arrayBuffer);
                
                // 2. Atualizar Memória Local
                knowledgeBase[file.name] = text;
                
                // 3. Salvar na Nuvem (Persistência)
                loadingItem.innerHTML = `<span>Salvando na nuvem...</span><i class="fas fa-cloud-upload-alt"></i>`;
                await saveKnowledgeToCloud();

                // 4. Atualizar UI
                renderDocsList();
                updateStatus('online', `Online | ${Object.keys(knowledgeBase).length} docs`);

            } catch (error) {
                console.error(error);
                loadingItem.innerHTML = `<span class="text-red-500">Erro!</span>`;
            }
        };

        async function extractTextFromPDF(arrayBuffer) {
            const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
            let text = "";
            for (let i = 1; i <= pdf.numPages; i++) {
                const page = await pdf.getPage(i);
                const content = await page.getTextContent();
                text += content.items.map(item => item.str).join(" ") + "\n";
            }
            return text;
        }

        // ================= CHAT =================

        window.sendMessage = async () => {
            const input = document.getElementById('user-input');
            const question = input.value.trim();
            if (!question) return;

            addMessage('user', question);
            input.value = '';
            const loadingId = addLoading();

            try {
                // Junta todo o texto de todos os documentos
                const allDocsText = Object.values(knowledgeBase).join("\n\n--- NOVO DOCUMENTO ---\n\n");
                
                const subject = document.getElementById('subject-select').value;
                let prompt = `Você é um tutor educacional chamado Teach AI.
                Matéria Atual: ${subject}.
                
                INSTRUÇÃO: Use APENAS o contexto abaixo. Se a resposta não estiver lá, diga que não sabe.
                
                CONTEXTO:
                ${allDocsText.substring(0, 900000)} 
                
                DÚVIDA: ${question}`;

                const result = await model.generateContent(prompt);
                const response = result.response;
                const text = response.text();

                removeLoading(loadingId);
                addMessage('ai', text);

            } catch (error) {
                removeLoading(loadingId);
                addMessage('ai', 'Erro ao conectar com a IA ou contexto muito grande.');
            }
        };

        // ================= UI HELPERS =================

        window.toggleScreen = (screenId) => {
            ['api-screen', 'dev-screen', 'chat-screen'].forEach(id => {
                document.getElementById(id).classList.add('hidden');
                document.getElementById(id).classList.remove('flex');
            });
            const target = document.getElementById(screenId);
            target.classList.remove('hidden');
            target.classList.add('flex');
        };

        window.saveApiKey = () => {
            const key = document.getElementById('api-key-input').value;
            if (key) {
                localStorage.setItem('gemini_key', key);
                initGemini(key);
                toggleScreen('chat-screen');
                if(supabase) loadKnowledgeFromCloud();
            }
        };

        window.resetChat = () => {
            document.getElementById('chat-box').innerHTML = '';
            addMessage('ai', 'Nova conversa iniciada. O que vamos estudar?');
        };

        function initGemini(apiKey) {
            genAI = new GoogleGenerativeAI(apiKey);
            model = genAI.getGenerativeModel({ model: "gemini-1.5-flash"});
        }

        function updateStatus(type, msg) {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');
            text.innerText = msg;
            if(type === 'loading') dot.className = "w-2 h-2 rounded-full bg-yellow-500 animate-pulse";
            else if(type === 'online') dot.className = "w-2 h-2 rounded-full bg-green-500";
            else dot.className = "w-2 h-2 rounded-full bg-red-500";
        }

        function addMessage(role, text) {
            const div = document.createElement('div');
            const isUser = role === 'user';
            div.className = `flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`;
            div.innerHTML = `
                <div class="w-8 h-8 rounded-full ${isUser ? 'bg-slate-300' : 'bg-indigo-600'} flex items-center justify-center flex-shrink-0">
                    <i class="fas ${isUser ? 'fa-user text-slate-600' : 'fa-robot text-white text-xs'}"></i>
                </div>
                <div class="p-4 rounded-2xl shadow-sm max-w-[85%] text-sm leading-relaxed ${
                    isUser ? 'bg-indigo-600 text-white rounded-tr-none' : 'bg-white text-slate-700 border border-slate-100 rounded-tl-none'
                }">
                    ${formatText(text)}
                </div>
            `;
            document.getElementById('chat-box').appendChild(div);
            document.getElementById('chat-box').scrollTop = document.getElementById('chat-box').scrollHeight;
        }

        function addLoading() {
            const id = 'loading-' + Date.now();
            const div = document.createElement('div');
            div.id = id;
            div.className = "flex gap-3";
            div.innerHTML = `
                <div class="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center flex-shrink-0">
                    <i class="fas fa-robot text-white text-xs"></i>
                </div>
                <div class="bg-white p-4 rounded-2xl rounded-tl-none shadow-sm border border-slate-100 flex gap-1 items-center">
                    <span class="text-xs text-slate-400 mr-2">Consultando documentos...</span>
                    <div class="w-2 h-2 bg-slate-400 rounded-full typing-dot"></div>
                    <div class="w-2 h-2 bg-slate-400 rounded-full typing-dot"></div>
                    <div class="w-2 h-2 bg-slate-400 rounded-full typing-dot"></div>
                </div>
            `;
            document.getElementById('chat-box').appendChild(div);
            document.getElementById('chat-box').scrollTop = document.getElementById('chat-box').scrollHeight;
            return id;
        }

        function removeLoading(id) { document.getElementById(id)?.remove(); }
        function formatText(text) { return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>'); }
    </script>
</body>
</html>
