import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
import time

# Add current path to find the best model
sys.path.append(os.getcwd())

def validate():
    print("--- FULL VALIDATION OF BEST INDIVIDUAL ---")
    
    # 1. Load the Best Model File
    try:
        from ab.gpt.brute.ga.modular.best_fractal_net import Net
        print("Successfully loaded 'Net' from best_fractal_net.py")
    except ImportError:
        print("Error: best_fractal_net.py not found or invalid.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 2. Setup Full CIFAR-10 (With Augmentation for real results)
    print("Loading Full CIFAR-10 Data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data_v2', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data_v2', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 3. Initialize Model
    # Parameters from your log (Gen 50 best)
    # 'drop_path_prob': 0.1, 'dropout': 0.1, 'lr': 0.01, 'momentum': 0.9
    prm = {'drop_path_prob': 0.1, 'dropout': 0.1, 'lr': 0.01, 'momentum': 0.9}
    
    model = Net(in_shape=(3, 32, 32), out_shape=(10,), prm=prm, device=device).to(device)
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # 4. Train for 20 Epochs
    EPOCHS = 20
    print(f"Starting Training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        start = time.time()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Test Acc: {acc:.2f}% | Time: {time.time()-start:.1f}s")

if __name__ == "__main__":
    validate()