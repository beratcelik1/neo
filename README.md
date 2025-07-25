# Smart Document Analyzer with AI Q&A

**AI-powered PDF document processing using Anthropic MCP + OpenAI GPT-4o-mini**

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=your_key_here" >> .env
```

### 2. Run (Pick One)

**💻 Terminal Interface (Direct)**
```bash
python terminal_interface.py
# No server needed - uses DocumentProcessor directly
```

**🌐 Web UI (Direct)**
```bash
cd webapp && python app.py
# No server needed - uses DocumentProcessor directly
# Open http://localhost:5001
```

**🔧 MCP Server + Client (MCP Protocol)**
```bash
# Terminal 1 - Start MCP Server
cd server && python smart_document_server.py

# Terminal 2 - Run MCP Client  
cd client && python document_client.py
```

**🔧 MCP Inspector (MCP Protocol)**
```bash
# Terminal 1 - Start MCP Server FIRST
python server/smart_document_server.py

# Terminal 2 - Connect Inspector
npx @modelcontextprotocol/inspector
# Connect: stdio + python + server/smart_document_server.py
```

## 📝 How to Use

**Upload & Search:**
```bash
# Terminal
upload test_documents/resume.pdf "My Resume"
search "programming languages"
ask "What skills does this person have?"
list

# Web UI: Just drag & drop PDFs and type questions
```

## 🏗️ What's Inside

```
neo/
├── .env                     # OpenAI API key
├── terminal_interface.py    # Command line interface  
├── server/
│   ├── document_processor.py    # AI engine (GPT + embeddings)
│   └── smart_document_server.py # MCP server
├── webapp/                  # Web interface
│   ├── app.py              
│   └── templates/index.html
├── data/                    # Your documents & vector DB
└── test_documents/          # Sample PDFs
```

## 🎯 Key Features

- **Real AI**: OpenAI GPT-4o-mini + embeddings (no mocks)
- **Smart Search**: Semantic vector search through documents
- **Multiple Interfaces**: Terminal, Web UI, MCP protocol
- **Balanced Chunking**: Optimized 1000-char chunks for better search
- **Production Ready**: Async, error handling, logging

## 🧹 Database Management

```bash
python cleanup_database.py
```

**Options:**
1. Show status
2. Remove duplicates  
3. Complete cleanup
4. Exit

## 🎤 Interview Demo

**2-minute demo flow:**

**Option A: Terminal (Simple)**
1. `python terminal_interface.py`
2. Upload: `upload test_documents/resume.pdf`
3. Search: `search "technical skills"`
4. Ask AI: `ask "What programming languages are mentioned?"`

**Option B: MCP Protocol (Technical)**
1. Terminal 1: `python server/smart_document_server.py`
2. Terminal 2: `python client/document_client.py`
3. Watch MCP protocol in action with real AI processing

**Option C: Web UI (Visual)**
1. `cd webapp && python app.py`
2. Open browser, drag & drop PDF, ask questions

**Talk about:**
- MCP protocol (Anthropic's standard)
- Server/client architecture
- Vector embeddings for semantic search
- Multiple interface options

## 🛠️ Troubleshooting

- **No API key**: Set `OPENAI_API_KEY` in `.env`
- **Port busy**: Web app finds available port automatically
- **Bad search**: Clean database with `python cleanup_database.py`
- **PDF errors**: Only PDFs supported currently