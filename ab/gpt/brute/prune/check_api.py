import sys
from pathlib import Path

# Dynamically add workspace roots to path based on script location
script_path = Path(__file__).resolve()
gw_root = script_path.parent.parent.parent.parent.parent  # Points to .../nn-gpt
ds_root = gw_root.parent / 'nn-dataset'          # Points to .../nn-dataset

sys.path.append(str(ds_root))
sys.path.append(str(gw_root))

import ab.nn.api as nn_dataset

try:
    print("Querying ResNet...")
    df = nn_dataset.data(nn='ResNet')
    if df.empty:
        print("DataFrame is empty!")
    else:
        print(f"Found {len(df)} rows.")
        code = df.iloc[0]['nn_code']
        print(f"Code length: {len(code)}")
        print("First 50 chars:", code[:50])
except Exception as e:
    print(f"API Error: {e}")
