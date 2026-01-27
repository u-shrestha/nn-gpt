import os
import re
import ast
import random
import subprocess
import textwrap
import shutil
import time
import json
from datetime import datetime

from ab.gpt.brute.ga.meta_evolution.llm_loader import LocalLLMLoader 
from ab.gpt.brute.ga.meta_evolution.rl_rewards import calculate_meta_reward

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_FILE = os.path.join(BASE_DIR, "genetic_algorithm.py")
RUNNER_SCRIPT = os.path.join(BASE_DIR, "run_fractal_evolution.py")
BACKUP_DIR = os.path.join(BASE_DIR, "ga_history_backup")
LOG_FILE = os.path.join(BASE_DIR, "meta_evolution_log.jsonl")
ADAPTER_SAVE_PATH = os.path.join(BASE_DIR, "fine_tuned_adapter")

# KEEP BENCHMARKS SMALL FOR FAST FEEDBACK
BENCH_GENS = 3
BENCH_POP = 10 

# --- MICRO-MUTATION PROMPT ---
PROMPTS = {
    "_mutate": """
You are an expert Python developer optimizing a Genetic Algorithm. 
Task: Modify the '_mutate' method to slightly improve exploration.

Constraints:
1. DO NOT change function arguments (self, chromosome).
2. Use 'self.search_space' and 'self.mutation_rate'.
3. CHANGE ONLY: The probability logic or how a new value is picked.
4. Keep the code simple and robust.

Input Code:
{code}

Output ONLY the python code starting with 'def _mutate(self, chromosome):'.
"""
}

class MetaEvolver:
    def __init__(self, model_path):
        self.llm = LocalLLMLoader(model_path, use_quantization=True, adapter_path=ADAPTER_SAVE_PATH)
        os.makedirs(BACKUP_DIR, exist_ok=True)
        
        print("[Meta] Running Baseline...")
        self.baseline_score = self.run_benchmark()
        print(f"[Meta] Baseline Score: {self.baseline_score:.4f}")

    def run_benchmark(self):
        cmd = ["python", RUNNER_SCRIPT, "--gens", str(BENCH_GENS), "--pop", str(BENCH_POP), "--clean"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            match = re.search(r"META_SCORE:\s*([\d\.]+)", res.stdout)
            return float(match.group(1)) if match else 0.0
        except: return 0.0

    def _extract_method(self, source_code, method_name):
        try:
            tree = ast.parse(source_code)
            t_class = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GeneticAlgorithm"), None)
            if not t_class: return None, None, 0
            node = next((n for n in t_class.body if isinstance(n, ast.FunctionDef) and n.name == method_name), None)
            if node:
                segment = ast.get_source_segment(source_code, node)
                start_idx = source_code.find(segment)
                return segment, (start_idx, start_idx + len(segment)), node.col_offset
        except: pass
        return None, None, 0

    def evolve_component(self, method_name):
        print(f"\n[Meta] Evolving: {method_name}")
        with open(TARGET_FILE, 'r') as f: full_code = f.read()
        
        orig_code, span, indent_col = self._extract_method(full_code, method_name)
        if not orig_code: return

        # LLM Generation
        prompt = PROMPTS[method_name].format(code=orig_code)
        raw_res = self.llm.generate(prompt, max_new_tokens=500)
        
        # Cleanup
        new_code = raw_res
        if "```python" in new_code: new_code = new_code.split("```python")[1].split("```")[0]
        elif "```" in new_code: new_code = new_code.split("```")[1].split("```")[0]
        new_code = textwrap.dedent(new_code).strip()
        
        # Indentation
        reindented = "\n".join([" " * indent_col + line for line in new_code.splitlines()])

        # Syntax Check & Injection
        valid_syntax = False
        try:
            test_full = full_code[:span[0]] + reindented + full_code[span[1]:]
            ast.parse(test_full)
            valid_syntax = True
        except SyntaxError as e:
            print(f"[Meta] Syntax Error: {e}")

        new_score = 0.0
        if valid_syntax:
            target_filename = os.path.basename(TARGET_FILE)
            bkp = os.path.join(BACKUP_DIR, f"{target_filename}_{method_name}.bak")
            shutil.copy(TARGET_FILE, bkp)
            with open(TARGET_FILE, 'w') as f: f.write(test_full)
            
            print("[Meta] Benchmarking...")
            new_score = self.run_benchmark()

        # RL Loop
        reward = calculate_meta_reward(new_score, self.baseline_score, valid_syntax)
        
        if reward > 0:
            print("--> SUCCESS. Updating Baseline & Fine-tuning.")
            self.baseline_score = new_score
            self.llm.train_on_buffer([{'prompt': prompt, 'completion': new_code}], epochs=1)
            self.llm.save_adapters(ADAPTER_SAVE_PATH)
        elif valid_syntax:
            print("--> Reverting File.")
            shutil.copy(bkp, TARGET_FILE)

if __name__ == "__main__":
    MODEL_PATH = "deepseek-ai/deepseek-coder-6.7b-instruct" 
    evolver = MetaEvolver(MODEL_PATH)
    
    # LOOP: FOCUS ONLY ON MUTATION FOR NOW
    for _ in range(10):
        evolver.evolve_component("_mutate") 
        time.sleep(2)