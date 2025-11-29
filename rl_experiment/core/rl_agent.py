# core/rl_agent.py
import numpy as np
import random
import json

class MutationAgent:
    def __init__(self, epsilon=0.2, alpha=0.1, gamma=0.9):
        """
        epsilon: Probability of choosing a random action (Explore)
        alpha: Learning rate (How much we accept new information)
        gamma: Discount factor (How much we care about future rewards)
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        # ACTIONS: The specific commands we will send to the LLM
        self.actions = [
            "add_conv_layer", 
            "increase_filters", 
            "decrease_filters",
            "add_dropout", 
            "remove_dropout",
            "add_batch_norm", 
            "change_activation",
            "add_residual_connection"
        ]
        
        # Q-Table: {(state, action): value}
        self.q_table = {}

    def get_state(self, train_acc, val_acc):
        """
        Discretizes the model metrics into a 'State'.
        State = (Performance Level, Overfitting Status)
        """
        # 1. Performance Level
        if val_acc < 0.20: perf = "very_low"
        elif val_acc < 0.50: perf = "low"
        elif val_acc < 0.75: perf = "medium"
        elif val_acc < 0.85: perf = "high"
        else: perf = "elite"

        # 2. Overfitting Status
        gap = train_acc - val_acc
        if gap > 0.15: status = "overfitting"
        elif gap < 0.02: status = "underfitting"
        else: status = "balanced"

        return (perf, status)

    def choose_action(self, state):
        # Initialize state in Q-table if not exists
        for action in self.actions:
            if (state, action) not in self.q_table:
                self.q_table[(state, action)] = 0.0

        # Epsilon-Greedy Strategy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions) # Explore
        else:
            # Exploit: Pick best action
            state_actions = {a: self.q_table[(state, a)] for a in self.actions}
            # Break ties randomly to avoid getting stuck
            max_val = max(state_actions.values())
            best_actions = [k for k, v in state_actions.items() if v == max_val]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        # Ensure next_state exists
        for a in self.actions:
            if (next_state, a) not in self.q_table:
                self.q_table[(next_state, a)] = 0.0

        # Q-Learning Formula
        current_q = self.q_table[(state, action)]
        max_next_q = max([self.q_table[(next_state, a)] for a in self.actions])
        
        new_q = current_q + self.alpha * (reward + (self.gamma * max_next_q) - current_q)
        self.q_table[(state, action)] = new_q

    def save_brain(self, filepath="q_table.json"):
        # Helper to save learning progress
        # Convert tuple keys to strings for JSON
        serializable_q = {str(k): v for k, v in self.q_table.items()}
        with open(filepath, 'w') as f:
            json.dump(serializable_q, f)