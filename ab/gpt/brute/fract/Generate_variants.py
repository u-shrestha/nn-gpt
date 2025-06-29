import os

TEMPLATE_PATH = "Fractal_template.py"
OUTPUT_DIR = "Files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(TEMPLATE_PATH, "r") as file:
    template = file.read()

counter = 1
for N in range(1, 11):  # 1 to 10
    for num_columns in range(1, 16):  # 1 to 15
        dropout = 0.3  # You can also randomize this if desired

        filled = (
            template
            .replace("?1", str(N))
            .replace("?2", str(num_columns))
            .replace("?3", str(dropout))
        )
    
        file_name = os.path.join(OUTPUT_DIR, f"Fractal_Net_{counter}.py")
        with open(file_name, "w") as f:
            f.write(filled)
        print(f"Generated: {file_name}")
        counter += 1
