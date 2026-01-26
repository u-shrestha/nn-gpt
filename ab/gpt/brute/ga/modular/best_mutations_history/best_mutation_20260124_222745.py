# Best Fitness: 0.40992
# Action: mutate_optimizer
# Date: 2026-01-24 22:30:48.460957

'''
 Prompt Used:
### Task: You are an expert. Rewrite the 'train_setup' method in Net class.
GOAL: Change the optimizer (e.g. use Adam, RMSprop, SGD with different momentum).
Input Code:
def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.01), 
            momentum=prm.get('momentum', 0.9)
        )
        return self.optimizer

    

History of previous attempts:
None

Based on the history, write BETTER code than before. Output ONLY the code block.
### Context (Original Code):
```python
def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.01), 
            momentum=prm.get('momentum', 0.9)
        )
        return self.optimizer

    
```
### Instructions:
1. Return ONLY the replacement code.
2. Do NOT explain.
3. Maintain indentation.
'''

# Generated Code:
def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.01), 
            momentum=prm.get('momentum', 0.9)
        )
        return self.optimizer

    

def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.01), 
            momentum=prm.get('momentum', 0.9)
        )
        return self.optimizer

    

def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.01), 
            momentum=prm.get('momentum', 0.9)
        )
        return self.optimizer

    

def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.01), 
            momentum=prm.get('momentum', 0.9)
        )
        return self.optimizer

    

def train_setup(self, prm):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=prm.get('lr', 0.01), 
            momentum=prm.get('momentum', 0.9)
        )
        return self.optim