import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets

# Define MLP model with different configurations
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define CNN model with different configurations
class CNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define transforms
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and create train and test datasets
train_set = datasets.USPS(root='./data', train=True, download=True, transform=transform)
test_set = datasets.USPS(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Define train function
def train_model(model, criterion, optimizer, train_loader, writer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        writer.add_scalar('Loss/Train', epoch_loss, epoch)

# Define evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return accuracy, precision, recall, conf_matrix

# Set up TensorBoard writer
writer = SummaryWriter()

# Initialize models, criterion, and optimizers with different configurations
mlp_model1 = MLP(28*28, 128, 64, 10)
mlp_model2 = MLP(28*28, 256, 128, 10)
mlp_model3 = MLP(28*28, 64, 32, 10)
cnn_model1 = CNN(1, 10)
cnn_model2 = CNN(1, 10)
cnn_model3 = CNN(1, 10)
criterion = nn.CrossEntropyLoss()
mlp_optimizer1 = optim.Adam(mlp_model1.parameters(), lr=0.001)
mlp_optimizer2 = optim.Adam(mlp_model2.parameters(), lr=0.001)
mlp_optimizer3 = optim.Adam(mlp_model3.parameters(), lr=0.001)
cnn_optimizer1 = optim.Adam(cnn_model1.parameters(), lr=0.001)
cnn_optimizer2 = optim.Adam(cnn_model2.parameters(), lr=0.001)
cnn_optimizer3 = optim.Adam(cnn_model3.parameters(), lr=0.001)

# Train and evaluate MLP models with different configurations
print("Training MLP Model 1...")
train_model(mlp_model1, criterion, mlp_optimizer1, train_loader, writer)
mlp_accuracy1, mlp_precision1, mlp_recall1, mlp_conf_matrix1 = evaluate_model(mlp_model1, test_loader)

print("Training MLP Model 2...")
train_model(mlp_model2, criterion, mlp_optimizer2, train_loader, writer)
mlp_accuracy2, mlp_precision2, mlp_recall2, mlp_conf_matrix2 = evaluate_model(mlp_model2, test_loader)

print("Training MLP Model 3...")
train_model(mlp_model3, criterion, mlp_optimizer3, train_loader, writer)
mlp_accuracy3, mlp_precision3, mlp_recall3, mlp_conf_matrix3 = evaluate_model(mlp_model3, test_loader)

# Train and evaluate CNN models with different configurations
print("Training CNN Model 1...")
train_model(cnn_model1, criterion, cnn_optimizer1, train_loader, writer)
cnn_accuracy1, cnn_precision1, cnn_recall1, cnn_conf_matrix1 = evaluate_model(cnn_model1, test_loader)

print("Training CNN Model 2...")
train_model(cnn_model2, criterion, cnn_optimizer2, train_loader, writer)
cnn_accuracy2, cnn_precision2, cnn_recall2, cnn_conf_matrix2 = evaluate_model(cnn_model2, test_loader)

print("Training CNN Model 3...")
train_model(cnn_model3, criterion, cnn_optimizer3, train_loader, writer)
cnn_accuracy3, cnn_precision3, cnn_recall3, cnn_conf_matrix3 = evaluate_model(cnn_model3, test_loader)

# Print results
print("\nMLP Model 1 Results:")
print(f"Accuracy: {mlp_accuracy1:.4f}")
print(f"Precision: {mlp_precision1:.4f}")
print(f"Recall: {mlp_recall1:.4f}")
print("Confusion Matrix:")
print(mlp_conf_matrix1)

print("\nMLP Model 2 Results:")
print(f"Accuracy: {mlp_accuracy2:.4f}")
print(f"Precision: {mlp_precision2:.4f}")
print(f"Recall: {mlp_recall2:.4f}")
print("Confusion Matrix:")
print(mlp_conf_matrix2)

print("\nMLP Model 3 Results:")
print(f"Accuracy: {mlp_accuracy3:.4f}")
print(f"Precision: {mlp_precision3:.4f}")
print(f"Recall: {mlp_recall3:.4f}")
print("Confusion Matrix:")
print(mlp_conf_matrix3)

print("\nCNN Model 1 Results:")
print(f"Accuracy: {cnn_accuracy1:.4f}")
print(f"Precision: {cnn_precision1:.4f}")
print(f"Recall: {cnn_recall1:.4f}")
print("Confusion Matrix:")
print(cnn_conf_matrix1)

print("\nCNN Model 2 Results:")
print(f"Accuracy: {cnn_accuracy2:.4f}")
print(f"Precision: {cnn_precision2:.4f}")
print(f"Recall: {cnn_recall2:.4f}")
print("Confusion Matrix:")
print(cnn_conf_matrix2)

print("\nCNN Model 3 Results:")
print(f"Accuracy: {cnn_accuracy3:.4f}")
print(f"Precision: {cnn_precision3:.4f}")
print(f"Recall: {cnn_recall3:.4f}")
print("Confusion Matrix:")
print(cnn_conf_matrix3)

# Close the writer
writer.close()
