import os
import torch
import random
import sys

# Ensure imports work from modular dir AND project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
# Adjusting 5 levels up: ab/gpt/brute/ga/modular -> ab/gpt/brute/ga -> ab/gpt/brute -> ab/gpt -> ab -> nn-gpt (root)
# wait, ab is in nn-gpt.
# ab/gpt/brute/ga/modular is 5 deep inside nn-gpt?
# /shared/ssd/home/b-a-singh/Thesis/nn-gpt (ROOT)
# /shared/ssd/home/b-a-singh/Thesis/nn-gpt/ab (Level 1)
# /shared/ssd/home/b-a-singh/Thesis/nn-gpt/ab/gpt (Level 2)
# /shared/ssd/home/b-a-singh/Thesis/nn-gpt/ab/gpt/brute (Level 3)
# /shared/ssd/home/b-a-singh/Thesis/nn-gpt/ab/gpt/brute/ga (Level 4)
# /shared/ssd/home/b-a-singh/Thesis/nn-gpt/ab/gpt/brute/ga/modular (Level 5)
# So dirname(current_dir) -> ga
# dirname(dirname) -> brute
# dirname(dirname(dirname)) -> gpt
# dirname(dirname(dirname(dirname))) -> ab
# dirname(dirname(dirname(dirname(dirname)))) -> nn-gpt (ROOT)
project_root = os.path.abspath(os.path.join(current_dir, "../../../../.."))

if current_dir not in sys.path:
    sys.path.append(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from .engine import GeneticAlgorithmEngine
from .selection import TournamentSelection
from .crossover import SinglePointCrossover
from .rl_mutation import RLLLMMutation
from .rl_rewards import evaluate_code_and_reward
from .mutation import MutationStrategy

# Load the seed code
try:
    seed_path = os.path.join(current_dir, "fractal_seed.py")
    with open(seed_path, "r") as f:
        SEED_CODE = f.read()
except FileNotFoundError:
    print(f"Error: {seed_path} not found. Run gen_fractal_script.py first.")
    exit(1)

# --- Configuration ---
POPULATION_SIZE = 4
GENERATIONS = 5
ELITISM_COUNT = 1
MUTATION_RATE = 0.8
MODEL_PATH = "deepseek-ai/deepseek-coder-1.3b-instruct"

# Define Search Space
SEARCH_SPACE = {
    "lr": [0.01, 0.001, 0.0005],
    "momentum": [0.9, 0.8],
    "drop_path_prob": [0.1, 0.2, 0.3],
    "dropout": [0.0, 0.1, 0.2]
}

def fitness_function(chromosome):
    """
    Fitness function wrapper.
    Checks for cached fitness from mutation first.
    """
    code = chromosome.get('code')
    if not code:
        return 0.0
    
    # Check cache (populated by RLLLMMutation)
    if chromosome.get('cached_fitness') is not None:
        print(f">>> Using Cached Fitness: {chromosome['cached_fitness']:.4f}")
        return chromosome['cached_fitness']
    
    # Evaluate if not cached (primary evaluation)
    print(">>> Evaluating Individual (No Cache)...")
    res = evaluate_code_and_reward(
        code,
        log_file=os.path.join(current_dir, "dataset/mutation_log.jsonl"),
        prompt_used="fitness_eval_no_cache" # Differentiate in logs
    )
    return res.get('val_metric', 0.0)

def main():
    print("--- Starting Single-Loop FractalNet Evolution ---")
    
    #Strategies
    selection = TournamentSelection(tournament_size=3)
    crossover = SinglePointCrossover()
    
    # LLM-Guided Mutation
    model_path = os.environ.get("LLM_MODEL_PATH", MODEL_PATH)
    print(f"Using LLM: {model_path}")
    
    mutation = RLLLMMutation(
        mutation_rate=MUTATION_RATE,
        model_path=model_path,
        use_quantization=False,
        log_file=os.path.join(current_dir, "dataset/mutation_log.jsonl")
    )

    engine = GeneticAlgorithmEngine(
        population_size=POPULATION_SIZE,
        search_space=SEARCH_SPACE,
        elitism_count=ELITISM_COUNT,
        selection_strategy=selection,
        crossover_strategy=crossover,
        mutation_strategy=mutation
    )

    # Initialize Population with Seed
    print("Injecting Seed into Population...")
    for ind in engine.population.individuals:
        ind.chromosome = {
            'code': SEED_CODE,
            'lr': 0.01, 'momentum': 0.9, 'drop_path_prob': 0.1, 'dropout': 0.1
        }
        ind.fitness = None

    # Run
    best_ind = engine.run(num_generations=GENERATIONS, fitness_function=fitness_function)
    
    print("--- Evolution Finished ---")
    print(f"Best Fitness: {best_ind.fitness}")
    
    # Save Best Code
    best_code_path = os.path.join(current_dir, "best_fractal_net.py")
    if best_ind.chromosome.get('code'):
        with open(best_code_path, "w") as f:
            f.write(best_ind.chromosome['code'])
        print(f"Saved best model to {best_code_path}")

if __name__ == "__main__":
    main()
