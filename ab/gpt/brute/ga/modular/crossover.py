from abc import ABC, abstractmethod
import random

class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1, parent2):
        pass

class SinglePointCrossover(CrossoverStrategy):
    def crossover(self, parent1, parent2):
        child_chromo = {}
        
        # Explicitly handle 'code' - inherit from one parent randomly
        if 'code' in parent1.chromosome and 'code' in parent2.chromosome:
            child_chromo['code'] = random.choice([parent1.chromosome['code'], parent2.chromosome['code']])
        elif 'code' in parent1.chromosome:
            child_chromo['code'] = parent1.chromosome['code']
        elif 'code' in parent2.chromosome:
            child_chromo['code'] = parent2.chromosome['code']

        # Crossover other hyperparameters
        genes = [k for k in parent1.chromosome.keys() if k != 'code']
        
        if len(genes) > 1:
            crossover_point = random.randint(1, len(genes) - 1)
        elif len(genes) == 1:
            crossover_point = 1
        else:
            crossover_point = 0

        for i, gene in enumerate(genes):
            if i < crossover_point:
                child_chromo[gene] = parent1.chromosome[gene]
            else:
                child_chromo[gene] = parent2.chromosome[gene]
        
        return child_chromo
