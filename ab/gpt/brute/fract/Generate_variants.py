import os

from ab.gpt.util.Const import synth_dir, epoch_dir, fract_dir

TEMPLATE_PATH = fract_dir / 'Fractal_template.py'
BASE_OUTPUT_DIR = synth_dir(epoch_dir(0))

# Ensure the base output directory exists
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Read the template file
with open(TEMPLATE_PATH, "r") as file:
    template = file.read()

counter = 0
for N in range(1, 11):  # 1 to 10
    for num_columns in range(1, 16):  # 1 to 15

        # Fill in the template
        filled = (
            template
            .replace("?1", str(N))
            .replace("?2", str(num_columns))
        )

        # Create model directory: out/nngpt/llm/epoch/A0/synth_nn/model_000/...
        model_id = "models"
        model_dir = os.path.join(BASE_OUTPUT_DIR, model_id)
        os.makedirs(model_dir, exist_ok=True)

        # Write the new_nn.py file
        model_path = os.path.join(model_dir, f"Fractal_Net_{counter}.py")
        with open(model_path, "w") as f:
            f.write(filled)

        print(f"Generated: {model_path}")
        counter += 1
