import requests
import json

def mutate_model(code, action):
    """
    Sends the code and the action to the local LLM (Ollama) to generate a mutated version.
    """
    
    prompt = f"""
    You are an expert AI researcher and Python coder.
    Your task is to modify the following PyTorch neural network code based on a specific instruction.
    
    INSTRUCTION: {action}
    
    RULES:
    1. Return ONLY the full valid Python code. No markdown, no explanations.
    2. Do not change the class name or the forward method signature.
    3. Ensure the code is runnable and imports are correct.
    4. Keep the same variable names for input/output.
    
    ORIGINAL CODE:
    {code}
    """
    
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3",  # Or any other model installed in Ollama
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        # Extract code from response (simple cleanup if LLM adds markdown)
        generated_text = result['response']
        
        # Basic cleanup to remove markdown code blocks if present
        if "```python" in generated_text:
            generated_text = generated_text.split("```python")[1].split("```")[0]
        elif "```" in generated_text:
            generated_text = generated_text.split("```")[1].split("```")[0]
            
        return generated_text.strip()
        
    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        # Return original code if mutation fails so we don't crash
        return code
