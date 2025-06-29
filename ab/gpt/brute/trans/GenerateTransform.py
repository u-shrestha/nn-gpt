import random
import os
import re
import torchvision.transforms as transforms
import torch
import itertools

# available transforms with parameter generators
variable_transforms = [
    {
        'name': 'CenterCrop',
        'params': lambda: {'size': random.randint(8, 64)}
    },
    {
        'name': 'Pad',
        'params': lambda: {
            'padding': random.randint(0, 10),
            'fill': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            'padding_mode': random.choice(['constant', 'edge', 'reflect', 'symmetric'])
        }
    },
    {
        'name': 'RandomApply',
        'params': lambda: {
            'transforms': [random.choice([
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2),
                transforms.RandomHorizontalFlip(p=0.5)
            ])],
            'p': round(random.uniform(0.1, 0.9), 2)
        }
    },
    {
        'name': 'RandomChoice',
        'params': lambda: {
            'transforms': [random.choice([
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2),
                transforms.RandomHorizontalFlip(p=0.5)
            ])]
        }
    },
    {
        'name': 'RandomOrder',
        'params': lambda: {
            'transforms': [random.choice([
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2),
                transforms.RandomHorizontalFlip(p=0.5)
            ]) for _ in range(3)]
        }
    },
    {
        'name': 'RandomCrop',
        'params': lambda: {'size': (random.randint(16, 64), random.randint(16, 64))}
    },
    {
        'name': 'RandomHorizontalFlip',
        'params': lambda: {'p': round(random.uniform(0.1, 0.9), 2)}
    },
    {
        'name': 'RandomVerticalFlip',
        'params': lambda: {'p': round(random.uniform(0.1, 0.9), 2)}
    },
    {
        'name': 'RandomResizedCrop',
        'params': lambda: {
            'size': random.randint(32, 64),
            'scale': (round(random.uniform(0.5, 0.8), 2), round(random.uniform(0.8, 1.0), 2)),
            'ratio': (round(random.uniform(0.75, 1.0), 2), round(random.uniform(1.0, 1.33), 2))
        }
    },
    {
        'name': 'FiveCrop',
        'params': lambda: {'size': random.randint(16, 64)}
    },
    {
        'name': 'TenCrop',
        'params': lambda: {'size': random.randint(16, 64)}
    },
    {
        'name': 'LinearTransformation',
        'params': lambda: {
            'transformation_matrix': torch.randn(3, 3) * 100  # Random matrix
        }
    },
    {
        'name': 'ColorJitter',
        'params': lambda: {
            'brightness': round(random.uniform(0.8, 1.2), 2),
            'contrast': round(random.uniform(0.8, 1.2), 2),
            'saturation': round(random.uniform(0.8, 1.2), 2),
            'hue': round(random.uniform(-0.1, 0.1), 2)
        }
    },
    {
        'name': 'RandomRotation',
        'params': lambda: {'degrees': random.randint(0, 30)}
    },
    {
        'name': 'RandomAffine',
        'params': lambda: {
            'degrees': random.randint(0, 30),
            'translate': (round(random.uniform(0.0, 0.2), 2), round(random.uniform(0.0, 0.2), 2)),
            'scale': (round(random.uniform(0.8, 1.2), 2), round(random.uniform(0.8, 1.2), 2)),
            'shear': round(random.uniform(-10, 10), 2)
        }
    },
    {
        'name': 'Grayscale',
        'params': lambda: {'num_output_channels': random.choice([1, 3])}
    },
    {
        'name': 'RandomGrayscale',
        'params': lambda: {'p': round(random.uniform(0.1, 0.9), 2)}
    },
    {
        'name': 'RandomPerspective',
        'params': lambda: {
            'distortion_scale': round(random.uniform(0.1, 0.5), 2),
            'p': round(random.uniform(0.1, 0.9), 2)
        }
    },
    {
        'name': 'RandomErasing',
        'params': lambda: {
            'p': round(random.uniform(0.1, 0.9), 2),
            'scale': (round(random.uniform(0.02, 0.33), 2), round(random.uniform(0.02, 0.33), 2)),
            'ratio': (round(random.uniform(0.3, 3.3), 2), round(random.uniform(0.3, 3.3), 2)),
            'value': random.choice(['random', (0, 0, 0)])
        }
    },
    {
        'name': 'GaussianBlur',
        'params': lambda: {
            'kernel_size': random.choice([3, 5, 7]),
            'sigma': (round(random.uniform(0.1, 2.0), 2), round(random.uniform(0.1, 2.0), 2))
        }
    },
    {
        'name': 'RandomInvert',
        'params': lambda: {'p': round(random.uniform(0.1, 0.9), 2)}
    },
    {
        'name': 'RandomPosterize',
        'params': lambda: {
            'bits': random.choice([4, 5, 6, 7, 8]),
            'p': round(random.uniform(0.1, 0.9), 2)
        }
    },
    {
        'name': 'RandomSolarize',
        'params': lambda: {
            'threshold': random.randint(0, 255),
            'p': round(random.uniform(0.1, 0.9), 2)
        }
    },
    {
        'name': 'RandomAdjustSharpness',
        'params': lambda: {
            'sharpness_factor': round(random.uniform(0.5, 2.0), 2),
            'p': round(random.uniform(0.1, 0.9), 2)
        }
    },
    {
        'name': 'RandomAutocontrast',
        'params': lambda: {'p': round(random.uniform(0.1, 0.9), 2)}
    },
    {
        'name': 'RandomEqualize',
        'params': lambda: {'p': round(random.uniform(0.1, 0.9), 2)}
    }
]

filename_counter = {}

def generate_transform_file(t1,t2, directory):
    
    # fixed transformations
    fixed_transforms = [
        'transforms.Resize((64,64))',
        'transforms.ToTensor()',
        'transforms.Normalize(*norm)'
    ]
    
    # truncate transform names to 5 letters
    name1 = t1['name']
    name2 = t2['name']
    base_name = f"{name1}_{name2}"

    # emove invalid characters
    base_name = re.sub(r'[^a-zA-Z0-9_]', '', base_name)

    # handle name clashes with counter
    if base_name in filename_counter:
        count = filename_counter[base_name] + 1
        filename_counter[base_name] = count
        output_filename = f"{base_name}_{count}.py"
    else:
        filename_counter[base_name] = 1
        output_filename = f"{base_name}.py"

    # ensure filename isn't too long
    max_filename_len = 250
    if len(output_filename) > max_filename_len:
        output_filename = output_filename[:max_filename_len - 4] + ".py"

    
    # generate transform code lines
    selected_transforms = []
    
    for vt in [t1,t2]:
        name = vt['name']
        params = vt['params']()
        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        
        selected_transforms.append(f"transforms.{name}({param_str})")
    
    all_transforms = selected_transforms + fixed_transforms

    transforms_code = ',\n    '.join(all_transforms)

    module_code = f"""import torch
import torchvision.transforms as transforms

def get_transform(norm):
    return transform = Compose([
    {transforms_code}
])
"""
   
    full_path = os.path.join(directory, output_filename)
    with open(full_path, 'w') as f:
        f.write(module_code)
        

    print(f"Saved transform to {full_path}")
    return full_path



# generate all permutations as a list
all_combinations = list(itertools.permutations(variable_transforms, 2))

# cycle from the permutations
all_combinations_cycle = itertools.cycle(all_combinations)

generate_num = 200
output_dir = 'transform_files'

# generate files by cycling through the permutations

for i in range(generate_num):
    t1, t2 = next(all_combinations_cycle)
    generate_transform_file(t1, t2, output_dir)