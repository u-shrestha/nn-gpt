import os
import shutil
import datetime

def archive_experiment():
    # 1. Create Timestamped Folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = f"output/experiments/exp_{timestamp}"
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Archiving experiment to: {dest_dir}")

    # 2. Define Files to Save
    files_to_save = [
        "best_model.py",
        "rl_q_table.json",
        "rl_ga_checkpoint.pkl",
        "dataset/mutation_results.jsonl",
        "main_rl_ga.py" # Save the script used
    ]

    # 3. Copy Files
    for file_path in files_to_save:
        if os.path.exists(file_path):
            # Maintain folder structure for logs
            dest_path = os.path.join(dest_dir, os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
            print(f"  Copied {file_path}")
        else:
            print(f"  Warning: {file_path} not found.")

    print("Archive complete.")

if __name__ == "__main__":
    archive_experiment()
