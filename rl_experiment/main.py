import os
import sys

# ---------------- CONFIGURATION ---------------- #
# Set FAST_MODE = True for debugging on your CPU (runs in seconds)
# Set FAST_MODE = False when running on the Cluster (runs for minutes/hours)
FAST_MODE = True  

GENERATIONS = 5
POPULATION_SIZE = 1  # Keep small for debugging
SEED_FILE = "population/seed_model.py"
# ----------------------------------------------- #

from core.rl_agent import MutationAgent
from core.llm_interface import mutate_model
from core.trainer import train_and_evaluate

def load_initial_population():
    """Reads the seed_model.py file to start the evolution."""
    if not os.path.exists(SEED_FILE):
        print(f"Error: Could not find {SEED_FILE}")
        sys.exit(1)
        
    with open(SEED_FILE, "r") as f:
        code = f.read()
    
    # Structure of an individual in the population
    return [{
        "id": "gen0_seed",
        "code": code, 
        "train_acc": 0.0, 
        "val_acc": 0.0
    }]

def main():
    print(f"--- Starting Evolution (Fast Mode: {FAST_MODE}) ---")
    
    # 1. Initialize the Brain (RL Agent)
    agent = MutationAgent()
    
    # 2. Load and Evaluate the Seed (Parent)
    population = load_initial_population()
    print("Evaluating Seed Model...")
    
    # Train the seed to get a baseline
    t_acc, v_acc = train_and_evaluate(population[0]["code"], fast_mode=FAST_MODE)
    population[0]["train_acc"] = t_acc
    population[0]["val_acc"] = v_acc
    
    best_model = population[0]
    print(f"Seed Baseline: Val Acc = {v_acc:.4f}")

    # 3. The Evolutionary Loop
    for gen in range(GENERATIONS):
        print(f"\n\n=== GENERATION {gen + 1} ===")
        
        next_gen_population = []
        
        for i, parent in enumerate(population):
            print(f"\n[Parent {i}] Acc: {parent['val_acc']:.4f}")
            
            # --- STEP A: RL DECIDES STRATEGY ---
            # Get current state (e.g., "Overfitting")
            state = agent.get_state(parent["train_acc"], parent["val_acc"])
            
            # Choose action (e.g., "add_dropout")
            action = agent.choose_action(state)
            print(f" -> Agent State: {state}")
            print(f" -> Agent Action: {action}")
            
            # --- STEP B: LLM WRITES CODE ---
            print(" -> Contacting LLM for mutation...")
            new_code = mutate_model(parent["code"], action)
            
            # --- STEP C: TRAIN NEW MODEL ---
            print(" -> Training Child Model...")
            child_t_acc, child_v_acc = train_and_evaluate(new_code, fast_mode=FAST_MODE)
            print(f" -> Result: Train={child_t_acc:.2f}, Val={child_v_acc:.2f}")

            # --- STEP D: REWARD & LEARN ---
            # Calculate Reward: Improvement in Validation Accuracy
            if child_v_acc == 0.0:
                reward = -2.0 # Penalty for broken code/crash
            else:
                reward = (child_v_acc - parent["val_acc"]) * 10
            
            print(f" -> Reward: {reward:.2f}")

            # Update the Q-Table (The Learning Step)
            next_state = agent.get_state(child_t_acc, child_v_acc)
            agent.update(state, action, reward, next_state)
            
            # --- STEP E: SURVIVAL ---
            # If the model works, add it to next generation
            if child_v_acc > 0:
                child = {
                    "id": f"gen{gen+1}_child{i}",
                    "code": new_code,
                    "train_acc": child_t_acc,
                    "val_acc": child_v_acc
                }
                next_gen_population.append(child)
                
                # Check if it's the new Best Global Model
                if child_v_acc > best_model["val_acc"]:
                    print(">>> NEW BEST MODEL FOUND! <<<")
                    best_model = child
                    # Save the python file
                    with open(f"population/best_model_gen{gen+1}.py", "w") as f:
                        f.write(new_code)

        # Simple Evolution Strategy:
        # If we found valid children, they become the parents for next gen.
        # If all children died (crashed), we retry with the old parents.
        if len(next_gen_population) > 0:
            population = next_gen_population
        else:
            print("Warning: All mutations failed this generation. Retrying with parents.")

    print("\n\n=== EVOLUTION FINISHED ===")
    print(f"Best Accuracy Achieved: {best_model['val_acc']:.4f}")
    print("RL Agent Q-Table (Brain):")
    print(agent.q_table)

if __name__ == "__main__":
    main()