from .mutation import MutationStrategy
from .llm_loader import LocalLLMLoader
from .rl_agent import RLAgent
from .rl_rewards import evaluate_code_and_reward
import random
import time

# Define a set of prompts (Actions)
PROMPTS = {
    "optimize_accuracy": (
        "You are an expert. Improve this PyTorch code to achieve higher accuracy. "
        "You can change layer sizes, kernel sizes, or activation functions. "
        "Keep the class name 'Net'."
    ),
    "deepen_network": (
        "You are an expert. Add one more Convolutional layer to this PyTorch model "
        "to increase its capacity. Ensure dimensions match. Keep the class name 'Net'."
    ),
    "add_skip_connection": (
        "You are an expert. Add a Residual Skip Connection to this PyTorch model "
        "if possible, or improve gradient flow. Keep the class name 'Net'."
    ),
    "reduce_parameters": (
        "You are an expert. Optimize this PyTorch code to be more efficient "
        "(fewer parameters) while maintaining accuracy. Keep the class name 'Net'."
    ),
    "fix_bugs": (
        "You are an expert. Review this code for any potential bugs or shape mismatches "
        "and fix them. Return the corrected code. Keep the class name 'Net'."
    )
}

class RLLLMMutation(MutationStrategy):
    def __init__(self, mutation_rate, model_path, use_quantization=True, 
                 q_table_path="q_table.json", log_file="dataset/mutation_log.jsonl"):
        super().__init__(mutation_rate)
        
        self.llm_loader = LocalLLMLoader(model_path, use_quantization)
        
        # Initialize RL Agent
        self.agent = RLAgent(
            actions=list(PROMPTS.keys()),
            q_table_path=q_table_path
        )
        
        self.log_file = log_file
        self.last_fitness = 0.0 # Track minimal state

    def get_state(self, fitness):
        """Discretize fitness into a state."""
        if fitness is None: return "fit_none"
        # Buckets: 0.0-0.2, 0.2-0.4, ...
        bucket = int(fitness * 5) 
        return f"fit_{bucket}"

    def mutate(self, chromosome, search_space):
        if random.random() > self.mutation_rate:
            return chromosome

        code = chromosome.get('code')
        if not code: return chromosome

        # 1. RL Agent chooses Action (Prompt)
        # We need a state. We assume chromosome has 'fitness' if it was evaluated, 
        # but pure chromosome dict usually doesn't have it. 
        # We might need to track it externally or pass Individual object.
        # For now, we use a simple globalish state tracking or random if unknown.
        # Ideally, 'chromosome' passed here is just the dict.
        current_state = "unknown" # Todo: get better state
        
        action_key = self.agent.choose_action(current_state)
        prompt_template = PROMPTS[action_key]
        
        full_prompt = (
            f"{prompt_template}\n\n"
            f"```python\n{code}\n```\n\n"
            "Provide ONLY the updated Python code."
        )

        try:
            print(f"[RL-GA] Action: {action_key}")
            new_code = self.llm_loader.generate(full_prompt)
            
            # Cleanup code
            if "```python" in new_code:
                new_code = new_code.split("```python")[1].split("```")[0].strip()
            elif "```" in new_code:
                new_code = new_code.split("```")[1].split("```")[0].strip()

            if new_code and "class Net" in new_code and "def forward" in new_code:
                
                # 2. Evaluate immediately to get Reward (for RL update)
                # Note: This is computationally expensive (double evaluation: here + GA loop)
                # But necessary for precise RL feedback on the *mutation step* itself.
                # Use quick evaluation (few batches).
                
                print("[RL-GA] Evaluating mutation reward...")
                res = evaluate_code_and_reward(
                    new_code,
                    log_file=self.log_file,
                    prompt_used=full_prompt, # Log for Fine-tuning
                    val_metric_baseline=0.1 # Should be previous fitness
                )
                
                reward = res.get('reward', 0.0)
                new_fitness = res.get('val_metric')
                if new_fitness is None: new_fitness = 0.0
                
                # 3. Update Agent
                next_state = self.get_state(new_fitness)
                self.agent.update(current_state, action_key, reward, next_state)
                self.agent.save()
                
                print(f"[RL-GA] Reward: {reward:.4f}, Acc: {new_fitness:.4f}")
                
                return {'code': new_code}
                
            else:
                print("[RL-GA] Invalid code generated.")
                # Penalty for invalid code?
                self.agent.update(current_state, action_key, -1.0, current_state)
                self.agent.save()

        except Exception as e:
            print(f"[RL-GA] Error: {e}")

        return chromosome
