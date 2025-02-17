import os
import time
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import ollama
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Configuration ---
# For the FastAPI endpoint, we use the DeepSeek R1 model via Hugging Face pipeline.
HF_MODEL = "deepseek-ai/DeepSeek-R1"
# For the file watcher we use ollama with a specified model tag (e.g. a 1.5B variant)
OLLAMA_MODEL = "deepseek-r1:1.5b"
# Directory to monitor for code changes
DIRECTORY = "./project"

# --- FastAPI Code Analysis Endpoint (using Hugging Face pipeline) ---
app = FastAPI(title="Code Analysis API")

class CodeInput(BaseModel):
    code: str

# Initialize the Hugging Face text-generation pipeline with DeepSeek R1.
generator = pipeline('text-generation', model=HF_MODEL)

def run_model(code: str):
    prompt = f"""
You are an AI code reviewer that analyzes Python code, detects errors, and suggests improvements.

### Code to review:
python
        {code}

### Task:
1. *Detect errors*: Identify syntax errors, incorrect function calls, and undefined variables.
2. *Explain issues*: Clearly state what is wrong.
3. *Suggest improvements*: Provide best practices or optimized solutions.
4. *Provide Corrected Code*: Share the corrected version.

### Expected Output:
- *Error analysis* (detailed explanation)
- *Improvement suggestions*
- *Corrected version of the code*
        """
    response = generator(prompt, max_length=500)
    analysis_text = response[0]['generated_text']
    return {"analysis": analysis_text}

@app.post("/analyze")
async def analyze_code_api(input: CodeInput):
    try:
        result = run_model(input.code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- File Monitoring and Analysis (using ollama) ---
def analyze_code_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # Get code review using ollama
    response = ollama.chat(model=OLLAMA_MODEL, messages=[
        {"role": "system", "content": "You are an expert code reviewer. Review the provided code, detect bugs, suggest improvements, and provide a fixed version."},
        {"role": "user", "content": f"Review this code and detect possible issues:\n\n{code}"}
    ])
    review = response['message']['content']

    # Get the fixed code version using ollama
    response_fixed = ollama.chat(model=OLLAMA_MODEL, messages=[
        {"role": "system", "content": "You are an expert code reviewer. Provide a corrected version of the given Python code."},
        {"role": "user", "content": f"Fix the following code:\n\n{code}"}
    ])
    fixed_code = response_fixed['message']['content']

    return review, fixed_code

def save_review_documentation(review, file_name):
    doc_path = os.path.join(DIRECTORY, "code_review_documentation.txt")
    with open(doc_path, 'a', encoding='utf-8') as doc_file:
        doc_file.write(f"Review for {file_name}:\n{review}\n{'-'*40}\n")
    print(f"Updated review for {file_name}")

def save_fixed_code(fixed_code, file_name):
    fixed_file_path = os.path.join(DIRECTORY, f"fixed_{file_name}")
    with open(fixed_file_path, 'w', encoding='utf-8') as fixed_file:
        fixed_file.write(fixed_code)
    print(f"Updated fixed code for {file_name}")

class CodeChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith(".py"):
            return
        file_name = os.path.basename(event.src_path)
        print(f"Detected change in {file_name}, analyzing...")
        review, fixed_code = analyze_code_file(event.src_path)
        save_review_documentation(review, file_name)
        save_fixed_code(fixed_code, file_name)

# --- Startup Event: Launch File Observer in a Background Thread ---
@app.on_event("startup")
async def start_file_observer():
    def run_observer():
        event_handler = CodeChangeHandler()
        observer = Observer()
        observer.schedule(event_handler, DIRECTORY, recursive=False)
        observer.start()
        print(f"Monitoring {DIRECTORY} for changes...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    thread = threading.Thread(target=run_observer, daemon=True)
    thread.start()

# --- Main: Run FastAPI Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
`    