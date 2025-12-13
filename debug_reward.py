import torch
from ab.gpt.brute.ga.modular.rl_rewards import evaluate_code_and_reward
import ab.gpt.brute.ga.modular.alexnet_mut as seed_module

# Read the seed code directly from the file we generated
with open("ab/gpt/brute/ga/modular/alexnet_mut.py", "r") as f:
    seed_code = f.read()

print("Debugging Reward Calculation for Seed Model...")
print("-" * 50)

# Run evaluation with full verbosity (we can't easily enable verbosity in the function, 
# but we can inspect the returned dict)
res = evaluate_code_and_reward(
    seed_code,
    val_metric_baseline=0.0
)

print("-" * 50)
print("FULL RESULT DICTIONARY:")
import json
# Custom serializer for things like functions/classes if any remain, though res should be simple types
print(res)

print("-" * 50)
if res.get("error"):
    print(f"ERROR DETECTED: {res['error']}")
else:
    print(f"Built OK: {res.get('built_ok')}")
    print(f"Forward OK: {res.get('forward_ok')}")
from ab.gpt.brute.ga.modular.rl_rewards import build_fn_from_code

print("-" * 50)
print("Testing Build ONLY:")
try:
    builder = build_fn_from_code(
        seed_code,
        in_shape=(2, 3, 32, 32),
        out_shape=(10,),
        prm={"lr": 0.01, "momentum": 0.9, "dropout": 0.5},
        device_str="cpu"
    )
    model = builder()
    print("Build Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Build Failed: {e}")
