import os
import random
from ab.gpt.brute.ga.modular import (
    GeneticAlgorithmEngine,
    TournamentSelection,
    CrossoverStrategy,
    Individual
)
from ab.gpt.brute.ga.modular.rl_mutation import RLLLMMutation
import torch.nn as nn

class IdentityCrossover(CrossoverStrategy):
    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            return parent1.chromosome
        return parent2.chromosome

def mock_fitness_ga(chromosome):
    # The real fitness evaluation happens inside RLLLMMutation for the RL reward,
    # but the GA engine also needs a fitness score. 
    # In a real scenario, we might want to cache it or re-evaluate.
    # For verification, we just return a placeholder or re-run a cheap check.
    code = chromosome.get('code', '')
    if not code: return 0.0
    try:
        # Very cheap syntax check
        compile(code, "<string>", "exec")
        return 0.5 # Default valid score
    except:
        return 0.0

def main():
    # Load Seed Code (AlexNet)
    seed_file = "ab/gpt/brute/ga/modular/alexnet_mut.py"
    if not os.path.exists(seed_file):
        seed_code = "import torch.nn as nn\nclass Net(nn.Module):\n    def forward(self, x): return x"
    else:
        with open(seed_file, 'r') as f:
            seed_code = f.read()

    # Initialize RL Mutation
    # We use a temp log file for verification
    log_file = "dataset/verify_log.jsonl"
    if os.path.exists(log_file): os.remove(log_file)
    
    mutation = RLLLMMutation(
        mutation_rate=1.0, # Force mutation to test RL loop
        model_path="ABrain/NNGPT-DeepSeek-Coder-1.3B-Instruct",
        use_quantization=True,
        q_table_path="verify_q_table.json",
        log_file=log_file
    )
    
    checkpoint_file = "verify_rl_checkpoint.pkl"
    if os.path.exists(checkpoint_file): os.remove(checkpoint_file)

    ga = GeneticAlgorithmEngine(
        population_size=2,
        search_space={},
        elitism_count=1,
        selection_strategy=TournamentSelection(tournament_size=2),
        crossover_strategy=IdentityCrossover(),
        mutation_strategy=mutation,
        checkpoint_path=checkpoint_file
    )

    # Initialize with Seed
    def seed_initialize(search_space):
        ga.population.individuals = [
            Individual({'code': seed_code}, fitness=0.1) 
            for _ in range(ga.population.size)
        ]
        print(f"Initialized population with seeded individuals.")
    ga.population.initialize = seed_initialize

    print("Starting RL-Guided GA Verification...")
    best = ga.run(num_generations=1, fitness_function=mock_fitness_ga)
    
    print("Evolution complete.")
    
    # Check if Q-table updated
    if os.path.exists("verify_q_table.json"):
        print("Success: Q-table created/updated.")
        with open("verify_q_table.json", 'r') as f:
            print("Q-Table content:", f.read())
    else:
        print("Failure: Q-table not found.")

    # Check if Log file created
    if os.path.exists(log_file):
        print("Success: Data log created.")
        with open(log_file, 'r') as f:
            print("Log content sample:", f.readline())
    else:
        print("Failure: Log file not found.")

if __name__ == "__main__":
    main()
