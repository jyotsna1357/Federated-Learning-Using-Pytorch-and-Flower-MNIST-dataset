import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score

class BasicBaseline:
    def __init__(self):
        # Initialize model, data loaders, and optimizer
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),  # Example input size for FashionMNIST
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 output classes
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_loader, self.test_loader = self.load_data()

    def load_data(self):
        # Load FashionMNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        return train_loader, test_loader

    def train(self, num_epochs, lr):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(self.train_loader)}')

    def test(self):
        self.model.eval()
        correct = 0
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True).view(-1)
                correct += pred.eq(target.view_as(pred)).sum().item()

                # Store predictions and true labels for metric calculation
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        # Calculate accuracy
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        # Calculate precision, recall, and F1-score
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f'Test Loss: {total_loss / len(self.test_loader):.4f}')
        print(f'Test Accuracy: {accuracy:.2f}%')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    def configure_attack(self, attack, malicious_clients):
        self.attack = attack
        self.malicious_clients = malicious_clients
        print(f"Configured attack: {attack.__class__.__name__} with {malicious_clients} malicious clients.")

    def configure_defense(self, defense):
        self.defense = defense
        print(f"Configured defense: {defense.__class__.__name__}.")

class FederatedBaseline:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.clients = [BasicBaseline() for _ in range(num_clients)]  # Initialize client models

    def load_data(self):
        # Optionally load data for each client if needed
        for client in self.clients:
            client.load_data()

    def train(self, num_epochs, rounds, lr):
        for round in range(rounds):
            print(f'Federated Round {round + 1}/{rounds}')
            for client in self.clients:
                client.train(num_epochs=num_epochs, lr=lr)

    def test(self):
        # Aggregate results from all clients
        overall_correct = 0
        all_preds = []
        all_labels = []
        
        for client in self.clients:
            client.model.eval()
            correct = 0
            total_loss = 0

            with torch.no_grad():
                for data, target in client.test_loader:
                    output = client.model(data)
                    loss = client.loss_fn(output, target)
                    total_loss += loss.item()

                    pred = output.argmax(dim=1, keepdim=True).view(-1)
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    # Store predictions and true labels for each client
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(target.cpu().numpy())
            
            accuracy = 100. * correct / len(client.test_loader.dataset)
            print(f'Client Test Accuracy: {accuracy:.2f}%')

        # Global metrics (across all clients)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f'Global Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    def configure_attack(self, attack, malicious_clients):
        self.attack = attack
        self.malicious_clients = malicious_clients
        print(f"Configured attack: {attack.__class__.__name__} with {malicious_clients} malicious clients.")

    def configure_defense(self, defense):
        self.defense = defense
        print(f"Configured defense: {defense.__class__.__name__}.")
