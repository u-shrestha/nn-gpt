from fractal_fn import fractal_fn
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_shape = (1, 3, 32, 32)
out_shape = (10,)
prm = {
    'lr': 0.01,
    'momentum': 0.9,
    'dropout': ?3,
    'N': ?1,
    'num_columns': ?2
}

model = fractal_fn(prm['N'], prm['num_columns'], prm['dropout'], in_shape, out_shape, device)

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), f"models/model_N{?1}_C{?2}.pt")