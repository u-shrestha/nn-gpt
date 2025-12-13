import random
from ab.gpt.brute.ga.modular import (
    GeneticAlgorithmEngine,
    TournamentSelection,
    SinglePointCrossover,
    RandomResettingMutation
)

def fitness_function(chromosome):
    # Simple fitness: sum of values
    return sum(chromosome.values())

def main():
    search_space = {
        'x': list(range(100)),
        'y': list(range(100)),
        'z': list(range(100))
    }
    
    selection = TournamentSelection(tournament_size=3)
    crossover = SinglePointCrossover()
    mutation = RandomResettingMutation(mutation_rate=0.1)
    
    ga = GeneticAlgorithmEngine(
        population_size=20,
        search_space=search_space,
        elitism_count=2,
        selection_strategy=selection,
        crossover_strategy=crossover,
        mutation_strategy=mutation,
        checkpoint_path='test_ga_checkpoint.pkl'
    )
    
    best = ga.run(num_generations=5, fitness_function=fitness_function)
    print(f"Best solution found: {best}")

if __name__ == "__main__":
    main()
