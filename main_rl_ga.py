import os
import argparse
from ab.gpt.brute.ga.modular import (
    GeneticAlgorithmEngine,
    TournamentSelection,
    CrossoverStrategy,
    Individual
)
from ab.gpt.brute.ga.modular.rl_mutation import RLLLMMutation

class IdentityCrossover(CrossoverStrategy):
    """
    Simple Crossover that just selects one parent.
    For complex code crossover, we might want AST-based merging later.
    """
    def crossover(self, parent1, parent2):
        # 50/50 chance to inherit from either parent
        import random
        if random.random() < 0.5:
            return parent1.chromosome
        return parent2.chromosome

def main():
    parser = argparse.ArgumentParser(description="Run RL-Guided Genetic Algorithm")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--pop_size", type=int, default=4, help="Population size")
    parser.add_argument("--mutation_rate", type=float, default=0.5, help="Mutation rate")
    parser.add_argument("--seed_file", type=str, default="ab/gpt/brute/ga/modular/fractal_mut.py", help="Path to seed code")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-6.7b-instruct", help="LLM Model path")
    args = parser.parse_args()

    # 1. Load Seed Code
    if not os.path.exists(args.seed_file):
        print(f"Seed file {args.seed_file} not found. Suggest running 'python3 ab/gpt/brute/ga/modular/gen_alexnet_script.py'")
        return
    with open(args.seed_file, 'r') as f:
        seed_code = f.read()

    # 2. Setup RL Mutation Strategy
    # This strategy handles: Prompt Selection (RL) -> Code Gen (LLM) -> Evaluation (Reward) -> Logging
    mutation = RLLLMMutation(
        mutation_rate=args.mutation_rate,
        model_path=args.model,
        use_quantization=True,
        q_table_path="rl_q_table.json",
        log_file="dataset/mutation_results.jsonl"
    )

    # 3. Setup GA Engine
    ga = GeneticAlgorithmEngine(
        population_size=args.pop_size,
        search_space={}, # Code mutation doesn't strictly use search_space dict
        elitism_count=1,
        selection_strategy=TournamentSelection(tournament_size=2),
        crossover_strategy=IdentityCrossover(),
        mutation_strategy=mutation,
        checkpoint_path="rl_ga_checkpoint.pkl"
    )

    # 4. Initialize Population with Seed
    # We override the default random initialization to use our seed code
    def seed_initialize(search_space):
        ga.population.individuals = [
            Individual({'code': seed_code}, fitness=None)  # Fitness None forces re-eval if needed
            for _ in range(ga.population.size)
        ]
        print(f"Initialized population of size {ga.population.size} with seed code.")
    
    ga.population.initialize = seed_initialize

    # 5. Define Fitness Function for the GA Loop
    # Note: RLLLMMutation *also* evaluates fitness internally to reward the agent.
    # To save time, we *should* ideally reuse that value. 
    # But current architecture separates them. 
    # For now, we reuse the same evaluation logic imported from rl_rewards.
    from ab.gpt.brute.ga.modular.rl_rewards import evaluate_code_and_reward
    
    def fitness_fn(chromosome):
        code = chromosome.get('code')
        if not code: return 0.0
        # Fast evaluation for the generation step
        res = evaluate_code_and_reward(code, val_metric_baseline=0.0)
        val = res.get('val_metric')
        return val if val is not None else 0.0

    # 6. Run Evolution
    print(f"Starting Evolution using {args.model}...")
    best = ga.run(num_generations=args.generations, fitness_function=fitness_fn)

    # 7. Results
    print("Evolution Finished.")
    print(f"Best Fitness: {best.fitness}")
    print(f"Best Code saved to 'best_model.py'")
    
    with open("best_model.py", "w") as f:
        f.write(best.chromosome['code'])

if __name__ == "__main__":
    main()
