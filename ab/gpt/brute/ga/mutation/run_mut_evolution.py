#run_mut_evolution.py:
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import json
from uuid import uuid4 as uuid4_orig
import time
import hashlib
from MutNet_evolvable import Net, SEARCH_SPACE, generate_model_code_string
from genetic_algorithm import GeneticAlgorithm

# Config
POPULATION_SIZE = 60
NUM_GENERATIONS = 50
MUTATION_RATE = 0.15
ELITISM_COUNT = 5
CHECKPOINT_FILE = 'ga_evolution_checkpoint.pkl'
BATCH_SIZE = 128
NUM_EPOCHS_PER_EVAL = 5

ARCHITECTURE_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'ga_architecture')
STATS_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'stats')
CHAMPION_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'ga-champ-mut.py')

os.makedirs(ARCHITECTURE_SAVE_DIR, exist_ok=True)
os.makedirs(STATS_SAVE_DIR, exist_ok=True)

seen_checksums = set()
architecture_counter = 0

def uuid4(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

def fitness_function(chromosome: dict) -> float:
    global architecture_counter

    if not any(chromosome.get(f'include_conv{i}', 0) for i in range(1, 6)):
        print("  - No conv layers included → fitness = 0")
        return 0.0

    try:
        model_code_string = generate_model_code_string(chromosome)
    except Exception as e:
        print(f"  - Code gen error: {e} → fitness = 0")
        return 0.0

    model_checksum = uuid4(model_code_string)
    if model_checksum in seen_checksums:
        print(f"  - Duplicate (checksum: {model_checksum[:8]}) → skip")
        return 0.0

    try:
        print(f"  - Evaluating unique arch (checksum: {model_checksum[:8]}...)")
        start_time = time.time()
        in_shape = (3, 32, 32)
        out_shape = (10,)
        model = Net(in_shape, out_shape, chromosome, device)
        model.train_setup(prm=chromosome)

        current_arch_number = architecture_counter
        model_base_name = f"ga-mut-{current_arch_number}"
        model_stats_dir_path = os.path.join(STATS_SAVE_DIR, f"img-classification_cifar-10_acc_{model_base_name}")
        os.makedirs(model_stats_dir_path, exist_ok=True)

        epoch_accuracies = []
        for epoch in range(NUM_EPOCHS_PER_EVAL):
            model.learn(train_loader)
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            epoch_accuracy = 100 * correct / total
            epoch_accuracies.append(epoch_accuracy)
            print(f"    - Epoch {epoch+1}/{NUM_EPOCHS_PER_EVAL} Accuracy: {epoch_accuracy:.2f}%")

            # ✅ FIX: Save as 1.json, 2.json, ..., NOT 0.json
            epoch_stats_filename = f"{epoch + 1}.json"
            epoch_stats_filepath = os.path.join(model_stats_dir_path, epoch_stats_filename)

            # Only include lr, momentum, dropout
            epoch_stats_data = {
                "accuracy": round(epoch_accuracy / 100.0, 4),
                "batch": BATCH_SIZE,
                "dropout": round(chromosome.get('dropout', 0.0), 4),
                "lr": round(chromosome.get('lr', 0.0), 4),
                "momentum": round(chromosome.get('momentum', 0.0), 4),
                "transform": "norm_256_flip",
                "uid": model_checksum,
            }
            try:
                with open(epoch_stats_filepath, 'w') as f:
                    json.dump([epoch_stats_data], f, indent=4)
            except Exception as e:
                print(f"      - Failed to save {epoch_stats_filepath}: {e}")

        final_accuracy = epoch_accuracies[-1] if epoch_accuracies else 0.0
        duration_ns = int((time.time() - start_time) * 1_000_000_000)

        # Save model code
        arch_filepath = os.path.join(ARCHITECTURE_SAVE_DIR, f"{model_base_name}.py")
        try:
            with open(arch_filepath, 'w') as f:
                f.write(model_code_string)
            print(f"  - Saved architecture to: {arch_filepath}")
            architecture_counter += 1
        except Exception as e:
            print(f"  - Failed to save arch: {e}")

        seen_checksums.add(model_checksum)
        return final_accuracy

    except Exception as e:
        print(f"  - Evaluation error: {e} → fitness = 0")
        return 0.0

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    full_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_subset, val_subset = random_split(full_train_set, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Input shape: (3, 32, 32), Output: 10 classes")

    # Resume architecture counter
    try:
        existing = [f for f in os.listdir(ARCHITECTURE_SAVE_DIR) if f.startswith("ga-mut-") and f.endswith(".py")]
        if existing:
            nums = []
            for f in existing:
                try:
                    num = int(f.replace("ga-mut-", "").replace(".py", ""))
                    nums.append(num)
                except:
                    pass
            if nums:
                architecture_counter = max(nums) + 1
                print(f"Resumed architecture counter: {architecture_counter}")
    except Exception as e:
        print(f"Could not resume counter: {e}")

    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        search_space=SEARCH_SPACE,
        elitism_count=ELITISM_COUNT,
        mutation_rate=MUTATION_RATE,
        checkpoint_path=CHECKPOINT_FILE
    )

    best = ga.run(num_generations=NUM_GENERATIONS, fitness_function=fitness_function)

    print("\n--- Evolution Complete ---")
    if best:
        print(f"Best fitness: {best['fitness']:.2f}%")
        champion_code = generate_model_code_string(best['chromosome'])
        with open(CHAMPION_SAVE_PATH, 'w') as f:
            f.write(champion_code)
        print(f"Champion saved to: {CHAMPION_SAVE_PATH}")
    else:
        print("No valid model found.")