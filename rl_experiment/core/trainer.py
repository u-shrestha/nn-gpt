import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
import io

def get_data_loaders(batch_size=64):
    """
    Standard CIFAR-10 data loader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # We download to './data'. This might take a minute the first time.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

def train_and_evaluate(model_code, epochs=1, fast_mode=True):
    """
    1. Compiles the string 'model_code' into a Python Class.
    2. Instantiates the model.
    3. Trains it.
    4. Returns Validation Accuracy.
    """
    # 1. Dynamic Code Execution
    # We create a dictionary to hold the executed code's variables
    local_scope = {}
    try:
        # EXECUTE THE CODE STRING
        exec(model_code, globals(), local_scope)
        
        # Instantiate the model (Assumes the class is named 'Net')
        if 'Net' not in local_scope:
            print("Error: The generated code did not define a class named 'Net'.")
            return 0.0, 0.0
            
        model = local_scope['Net']()
        
        # Check if model runs on a dummy input
        dummy_input = torch.randn(1, 3, 32, 32)
        _ = model(dummy_input)
        
    except Exception as e:
        print(f"Compilation/Shape Error: {e}")
        return 0.0, 0.0 # Fitness 0 if broken

    # 2. Setup Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    trainloader, testloader = get_data_loaders()
    
    # 3. Training Loop
    print(f"Training on {device}...")
    model.train()
    
    # LIMITER FOR CPU DEBUGGING
    max_batches = 5 if fast_mode else len(trainloader)
    
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if i >= max_batches: break # Stop early if fast_mode
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Track Train Acc
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_acc = correct / total

    # 4. Validation Loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            if i >= max_batches: break # Stop early if fast_mode
            
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f"Result: Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.2f}")
    
    return train_acc, val_acc