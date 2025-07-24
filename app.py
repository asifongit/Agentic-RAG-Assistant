from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from crewai_tools import RagTool
from crewai import Agent, Task, Crew, LLM

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.getenv("SECRET_KEY", "a_secret_key_for_flash_messages") # Add a secret key for flash messages

# Track which PDFs were processed
processed_files = set()
active_file = None

# RAG configuration
config = {
    "llm": {
        "provider": "groq",
        "config": {
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "api_key": os.getenv("GROQ_API_KEY"),
        }
    },
    "embedding_model": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-mpnet-base-v2"
        }
    },
    "vectordb": {
        "provider": "chroma",
        "config": {
            "collection_name": "my-collection"
        }
    },
    "chunker": {
        "chunk_size": 400,
        "chunk_overlap": 100,
        "length_function": "len",
        "min_chunk_size": 200
    }
}

@app.route('/', methods=['GET'])
def index():
    # We can pass messages via flash instead of directly as 'answer'
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global active_file

    pdf_file = request.files.get('pdf')
    if not pdf_file or pdf_file.filename == '':
        flash("❌ Please upload a PDF.", "error")
        return redirect(url_for('index'))

    filename = secure_filename(pdf_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if filename not in processed_files:
        pdf_file.save(filepath)
        rag_tool = RagTool(config=config, summarize=True)
        rag_tool.add(data_type="pdf_file", source=filepath)
        processed_files.add(filename)

    active_file = filename
    flash(f"✅ {filename} uploaded and embedded successfully.", "success")
    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask_question():
    global active_file

    question = request.form.get('question')
    if not question:
        flash("❌ Please enter a question.", "error")
        return redirect(url_for('index'))
    if not active_file:
        flash("❌ No PDF is currently active. Please upload one first.", "error")
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], active_file)
    rag_tool = RagTool(config=config, summarize=True)
    rag_tool.add(data_type="pdf_file", source=filepath)

    llm = LLM(model="groq/meta-llama/llama-4-scout-17b-16e-instruct")
    agent = Agent(
        role="Knowledge Expert",
        goal="Answer questions accurately using document knowledge",
        backstory="You're a document-savvy assistant who retrieves facts.",
        tools=[rag_tool],
        llm=llm,
        verbose=False
    )

    task = Task(
        description=f"Answer the question: {question}",
        expected_output="A helpful answer based on the document.",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )
    answer = crew.kickoff()

    flash(f"🧠 Answer: {answer}", "info")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)