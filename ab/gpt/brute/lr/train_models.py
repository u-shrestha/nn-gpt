"""Fine-tuning script for ResNet18 on CIFAR-10"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import time


class CIFARTrainer:
    def __init__(self, model_name='ResNet18', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Dataset setup
        self.train_loader = None
        self.test_loader = None
        self._setup_data()
        
    def _setup_data(self):
        """Load CIFAR-10 dataset"""
        print(f"Loading CIFAR-10 dataset...")
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=2
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=2
        )
        
        print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    def _load_model(self):
        """Load ResNet18 model"""
        print(f"Loading {self.model_name}...")
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, 10)  # CIFAR-10 has 10 classes
        self.model = self.model.to(self.device)
        print(f"✓ Model loaded: {self.model_name}")
        
    def train_epoch(self, optimizer, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{'~' + str(len(self.train_loader))}: "
                      f"Loss={loss.item():.4f}, Acc={100. * correct / total:.2f}%")
        
        if scheduler:
            scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def test(self):
        """Test the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        accuracy = 100. * correct / total
        return accuracy
    
    def train(self, epochs=10, lr=0.01, scheduler_type='step'):
        """Complete training loop"""
        self._load_model()
        
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        # Setup scheduler
        scheduler = None
        if scheduler_type == 'step':
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == 'exponential':
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        
        print(f"\nTraining {self.model_name} for {epochs} epochs")
        print(f"Learning Rate: {lr}, Scheduler: {scheduler_type}")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        results = {
            'model': self.model_name,
            'scheduler': scheduler_type,
            'epochs': epochs,
            'lr': lr,
            'train_losses': [],
            'train_accs': [],
            'test_accs': []
        }
        
        start_time = time.time()
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(optimizer, scheduler)
            test_acc = self.test()
            
            results['train_losses'].append(train_loss)
            results['train_accs'].append(train_acc)
            results['test_accs'].append(test_acc)
            
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Acc:  {test_acc:.2f}%")
            print()
        
        elapsed_time = time.time() - start_time
        results['training_time'] = elapsed_time
        
        print(f"Training completed in {elapsed_time / 60:.2f} minutes")
        print(f"Best Test Accuracy: {max(results['test_accs']):.2f}%")
        
        return results


def main():
    """Main fine-tuning execution"""
    print("\n" + "=" * 70)
    print("ResNet18 Fine-Tuning on CIFAR-10")
    print("=" * 70)
    
    # Initialize database
    print("\nInitializing database...")
    init_database()
    print("✓ Database initialized")
    
    # Configuration
    model_name = ACTIVE_MODEL  # ResNet18
    epochs = 10
    lr = 0.01
    schedulers = ['step', 'cosine', 'exponential']
    
    # Train with different schedulers
    all_results = []
    
    for scheduler_type in schedulers:
        print(f"\n{'=' * 70}")
        print(f"Training with {scheduler_type.upper()} scheduler")
        print(f"{'=' * 70}")
        
        trainer = CIFARTrainer(model_name=model_name)
        results = trainer.train(epochs=epochs, lr=lr, scheduler_type=scheduler_type)
        
        # Save results to database
        print(f"\nSaving results to database...")
        save_scheduler_result(
            model_name=model_name,
            scheduler_type=scheduler_type,
            epoch=epochs,
            learning_rate=lr,
            best_accuracy=max(results['test_accs']),
            final_accuracy=results['test_accs'][-1],
            training_time=results['training_time']
        )
        print("✓ Results saved")
        
        all_results.append(results)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 70}")
    for results in all_results:
        best_acc = max(results['test_accs'])
        final_acc = results['test_accs'][-1]
        print(f"\n{results['scheduler'].upper()} Scheduler:")
        print(f"  Best Accuracy:  {best_acc:.2f}%")
        print(f"  Final Accuracy: {final_acc:.2f}%")
        print(f"  Training Time:  {results['training_time'] / 60:.2f} minutes")
    
    print("\n✓ Fine-tuning completed successfully!")


if __name__ == '__main__':
    main()
