import os
import argparse
import hashlib
from genetic_algorithm import GeneticAlgorithm
from FractalNet_evolvable import SEARCH_SPACE, generate_model_code_string
from ab.gpt.util.Eval import Eval

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# This is the folder where unique fractal models will be saved
ARCH_DIR = os.path.join(BASE_DIR, 'ga_fractal_arch') # Keep for execution
FRACTALS_DIR = os.path.join(BASE_DIR, 'Fractals')    # New permanent storage
STATS_DIR = os.path.join(BASE_DIR, 'stats')
CHECKPOINT = 'fractal_ga_ckpt.pkl'

os.makedirs(ARCH_DIR, exist_ok=True)
os.makedirs(FRACTALS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

seen_hashes = set()

def get_hash(s): return hashlib.md5(s.encode()).hexdigest()

def fitness_function(chromosome):
    try:
        # 1. Generate Source Code
        code_str = generate_model_code_string(chromosome)
        code_hash = get_hash(code_str)
        
        # Deduplication
        if code_hash in seen_hashes: return 0.0
        
        # 2. Save Model File to ARCH_DIR
        model_name = f"fractal_ga_{code_hash[:8]}"
        filepath = os.path.join(ARCH_DIR, f"{model_name}.py")
        
        # Save to execution dir
        with open(filepath, 'w') as f: 
            f.write(code_str)
            
        # Save copy to Fractals folder for user inspection
        fractal_path = os.path.join(FRACTALS_DIR, f"{model_name}.py")
        with open(fractal_path, 'w') as f: f.write(code_str)
            
        # 3. Evaluate
        eval_prm = {
            'lr': chromosome['lr'],
            'momentum': chromosome['momentum'],
            'batch': 64, 
            'epoch': 2, # Short epochs for Meta-Evaluation
            'transform': "norm_256_flip" 
        }
        
        stats_path = os.path.join(STATS_DIR, f"stats_{model_name}")
        
        evaluator = Eval(
            model_source_package=ARCH_DIR,
            task='img-classification',
            dataset='cifar-10',
            metric='acc',
            prm=eval_prm,
            save_to_db=False,
            prefix=model_name,
            save_path=stats_path
        )
        
        res = evaluator.evaluate(filepath)
        
        acc = 0.0
        if isinstance(res, dict): acc = res.get('accuracy', 0.0)
        elif isinstance(res, (float, int)): acc = float(res)
        elif isinstance(res, tuple): acc = float(res[1])
        
        seen_hashes.add(code_hash)
        # Store accuracy in chromosome for later "best" retrieval
        chromosome['accuracy'] = acc * 100
        return acc * 100 
        
    except Exception:
        # Uncomment for debugging: print(f"Eval Fail: {e}")
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", type=int, default=3)
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    try:
        ga = GeneticAlgorithm(
            population_size=args.pop,
            search_space=SEARCH_SPACE,
            elitism_count=2,
            mutation_rate=0.2,
            checkpoint_path=CHECKPOINT
        )
        
        best, history = ga.run(args.gens, fitness_function)
        
        # Save Best Architecture
        if best:
             best_code = generate_model_code_string(best['chromosome'])
             best_path = os.path.join(BASE_DIR, "best_fractal_model.py")
             with open(best_path, "w") as f:
                 f.write(best_code)
             # print(f"Saved best model to {best_path}") # Keep silent for meta-score parsing

        # Meta-Score Calculation
        if history:
            avg_imp = (history[-1] - history[0]) if len(history) > 1 else 0
            peak = max(history)
            meta_score = peak + (avg_imp * 1.5) 
        else:
            meta_score = 0.0
            
        print(f"META_SCORE: {meta_score:.4f}")

    except Exception:
        # print(f"CRITICAL GA FAIL: {e}")
        print("META_SCORE: 0.0")