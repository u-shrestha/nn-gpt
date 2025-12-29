import os
import sys
import random

# --- Import Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../.."))

if current_dir not in sys.path:
    sys.path.append(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Module Imports ---
from .engine import GeneticAlgorithmEngine
from .selection import TournamentSelection
from .crossover import SinglePointCrossover
from .rl_mutation import LLMMutationStrategy 
from .rl_rewards import evaluate_fitness

# --- Load Seed ---
try:
    seed_path = os.path.join(current_dir, "fractal_seed.py")
    with open(seed_path, "r") as f:
        SEED_CODE = f.read()
except FileNotFoundError:
    print(f"Error: {seed_path} not found. Run gen_fractal_script.py first.")
    exit(1)

# --- Custom Crossover Strategy (CRITICAL FIX) ---
class CodePreservingCrossover(SinglePointCrossover):
    """
    Standard crossover mixes hyperparameters, but destroys 'code' if not handled.
    This custom class ensures the child inherits code from one of the parents.
    """
    def crossover(self, parent1, parent2):
        # 1. Perform standard hyperparameter crossover
        child1_chrom, child2_chrom = super().crossover(parent1, parent2)
        
        # 2. Explicitly copy code from parents
        # Strategy: Child 1 gets Parent 1's code, Child 2 gets Parent 2's code
        # (Mutation will change this code later)
        p1_code = parent1.chromosome.get('code', SEED_CODE)
        p2_code = parent2.chromosome.get('code', SEED_CODE)
        
        child1_chrom['code'] = p1_code
        child2_chrom['code'] = p2_code
        
        return child1_chrom, child2_chrom

# --- Configuration ---
POPULATION_SIZE = 4
GENERATIONS = 5
ELITISM_COUNT = 1
USE_QUANTIZATION = True 
MUTATION_RATE = 0.5 
MODEL_PATH = "deepseek-ai/deepseek-coder-1.3b-instruct"

# Search Space (Hyperparameters only)
SEARCH_SPACE = {
    "lr": [0.01, 0.005, 0.001],
    "momentum": [0.9, 0.8],
    "drop_path_prob": [0.1, 0.2],
    "dropout": [0.0, 0.1]
}

# --- Glue Code ---
class IndividualWrapper:
    def __init__(self, chromosome):
        self.code = chromosome.get('code', '')
        self.prm = chromosome

def fitness_function(chromosome):
    """
    Evaluation with a 'Safety Net'.
    If the engine wiped the seed, we put it back right here.
    """
    # SAFETY NET: If code is missing (due to random init), inject the Seed!
    if 'code' not in chromosome or not chromosome['code']:
        # print(">>> [System] Re-injecting Seed Code into empty individual")
        chromosome['code'] = SEED_CODE

    # Check cache
    if chromosome.get('cached_fitness') is not None:
        print(f">>> Using Cached Fitness: {chromosome['cached_fitness']:.4f}")
        return chromosome['cached_fitness']
    
    wrapper = IndividualWrapper(chromosome)
    
    # print(">>> Evaluating Individual...")
    accuracy = evaluate_fitness(wrapper)
    
    return accuracy

def main():
    print("--- Starting Single-Loop FractalNet Evolution ---")
    
    # 1. Strategies
    selection = TournamentSelection(tournament_size=3)
    
    # FIX: Use our Custom Crossover
    crossover = CodePreservingCrossover() 
    
    print(f"Using LLM: {MODEL_PATH} (Quantized: {USE_QUANTIZATION})")
    mutation = LLMMutationStrategy(
        model_path=MODEL_PATH,
        mutation_rate=MUTATION_RATE,
    )

    engine = GeneticAlgorithmEngine(
        population_size=POPULATION_SIZE,
        search_space=SEARCH_SPACE,
        elitism_count=ELITISM_COUNT,
        selection_strategy=selection,
        crossover_strategy=crossover, # Pass custom class
        mutation_strategy=mutation
    )

    # 2. Run Evolution
    # Note: We rely on fitness_function to inject the seed dynamically now,
    # so we don't need to manually inject before run() if run() wipes it anyway.
    best_ind = engine.run(num_generations=GENERATIONS, fitness_function=fitness_function)
    
    print("--- Evolution Finished ---")
    print(f"Best Fitness: {best_ind.fitness}")
    
    # 3. Save Results
    best_code_path = os.path.join(current_dir, "best_fractal_net.py")
    if best_ind.chromosome.get('code'):
        with open(best_code_path, "w") as f:
            f.write(best_ind.chromosome['code'])
        print(f"Saved best model to {best_code_path}")

if __name__ == "__main__":
    main()