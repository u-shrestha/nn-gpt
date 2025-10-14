#genetic_algorithm.py
import random
from tqdm import tqdm
import pickle
import os

class GeneticAlgorithm:
    def __init__(self, population_size, search_space, elitism_count, mutation_rate,
                 checkpoint_path='ga_checkpoint.pkl'):
        self.population_size = population_size
        self.search_space = search_space
        self.elitism_count = elitism_count
        self.mutation_rate = mutation_rate
        self.population = []
        self.checkpoint_path = checkpoint_path

    def _create_random_chromosome(self):
        chromosome = {}
        for key, values in self.search_space.items():
            chromosome[key] = random.choice(values)
        return chromosome

    def _initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            chromosome = self._create_random_chromosome()
            self.population.append({'chromosome': chromosome, 'fitness': None})
        print(f"Initialized population with {self.population_size} individuals.")

    def _save_checkpoint(self, generation_num):
        state = {
            'generation': generation_num,
            'population': self.population
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"--- Checkpoint saved for Generation {generation_num} ---")

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, 'rb') as f:
                state = pickle.load(f)
            print(f"--- Resuming from checkpoint at Generation {state['generation']} ---")
            return state['generation'], state['population']
        return 0, None

    def _crossover(self, parent1_chromo, parent2_chromo):
        child_chromo = {}
        genes = list(self.search_space.keys())
        # Single-point crossover
        crossover_point = random.randint(1, len(genes) - 1)
        for i, gene in enumerate(genes):
            if i < crossover_point:
                child_chromo[gene] = parent1_chromo[gene]
            else:
                child_chromo[gene] = parent2_chromo[gene]
        return child_chromo

    def _mutate(self, chromosome):
        mutated_chromo = chromosome.copy()
        for gene in self.search_space.keys():
            if random.random() < self.mutation_rate:
                current_value = mutated_chromo[gene]
                possible_values = self.search_space[gene]
                # Exclude current value if possible
                other_values = [v for v in possible_values if v != current_value]
                if other_values:
                    mutated_chromo[gene] = random.choice(other_values)
                else:
                    # Only one option – keep as is
                    pass
        return mutated_chromo

    def _selection(self):
        # Tournament selection (size = 3)
        tournament_size = min(3, len(self.population))
        competitors = random.sample(self.population, tournament_size)
        winner = max(competitors, key=lambda x: x['fitness'] if x['fitness'] is not None else -1)
        return winner

    def run(self, num_generations, fitness_function):
        start_gen, loaded_population = self._load_checkpoint()
        if loaded_population is not None:
            self.population = loaded_population
        else:
            self._initialize_population()

        best_ever_individual = None

        # If resuming, find current best
        if loaded_population is not None:
            evaluated = [ind for ind in self.population if ind['fitness'] is not None]
            if evaluated:
                best_ever_individual = max(evaluated, key=lambda x: x['fitness'])

        for gen in range(start_gen, num_generations):
            print(f"\n===== Generation {gen + 1}/{num_generations} =====")
            print("Evaluating fitness of population...")

            for individual in tqdm(self.population):
                if individual['fitness'] is None:
                    individual['fitness'] = fitness_function(individual['chromosome'])

            # Sort by fitness (descending)
            self.population.sort(key=lambda x: x['fitness'] if x['fitness'] is not None else -1, reverse=True)

            # Track best ever
            current_best = self.population[0]
            if best_ever_individual is None or (
                current_best['fitness'] is not None and
                (best_ever_individual['fitness'] is None or current_best['fitness'] > best_ever_individual['fitness'])
            ):
                best_ever_individual = current_best.copy()
                print(f"*** New best overall fitness: {best_ever_individual['fitness']:.4f} ***")

            best_fitness = current_best['fitness']
            print(f"Best fitness in Generation {gen + 1}: {best_fitness:.4f}")
            print(f"Best chromosome: {current_best['chromosome']}")

            # Build next generation
            next_generation = []

            # Elitism
            if self.elitism_count > 0:
                elites = self.population[:self.elitism_count]
                next_generation.extend(elites)
                print(f"Carried over {len(elites)} elite(s).")

            # Generate offspring
            num_children = self.population_size - len(next_generation)
            if num_children > 0:
                print(f"Creating {num_children} offspring via crossover + mutation...")
                for _ in range(num_children):
                    parent1 = self._selection()
                    parent2 = self._selection()
                    child_chromo = self._crossover(parent1['chromosome'], parent2['chromosome'])
                    mutated_child = self._mutate(child_chromo)
                    next_generation.append({'chromosome': mutated_child, 'fitness': None})

            self.population = next_generation
            self._save_checkpoint(generation_num=gen + 1)

        print("\n===== Evolution Complete =====")
        return best_ever_individual