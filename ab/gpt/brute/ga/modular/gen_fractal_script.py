import shutil
import os

source = 'ab/gpt/brute/ga/modular/fractalnet.py'
dest = 'ab/gpt/brute/ga/modular/fractal_mut.py'

print(f"Generating {dest} from {source}...")

# Read base code
with open(source, 'r') as f:
    code = f.read()

# Append initial chromosome metadata
# This matches the parameters in fractalnet.py Net.__init__
chromosome_str = """
# Chromosome used to generate this model:
# {'lr': 0.01, 'momentum': 0.9, 'dropout': 0.0, 'drop_path_prob': 0.1, 'recursion_depth': 3, 'columns': 2, 'base_channels': 64}
"""

with open(dest, 'w') as f:
    f.write(code + chromosome_str)

print("Done. FractalNet seed ready.")
