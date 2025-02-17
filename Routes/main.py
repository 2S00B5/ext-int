from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Initialize the text-generation pipeline.
# Ensure that 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' is available and properly installed.
generator = pipeline('text-generation', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')

app = FastAPI(title="Code Analysis API")

class CodeInput(BaseModel):
    code: str

def run_model(code: str):
    # Build the prompt by inserting the user-provided code into the template.
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
async def analyze_code(input: CodeInput):
    try:
        result = run_model(input.code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    