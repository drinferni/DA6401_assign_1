import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set print options so it doesn't truncate the output
torch.set_printoptions(threshold=float('inf'), linewidth=200, precision=4, sci_mode=False)

# 1. Configuration
params = {
    "learning_rate": 0.5,
    "num_hidden_layer": 2,
    "size_of_hidden_layer": 2,
    "epochs": 1,
    "batch_size": 60000,
    "input_size": 784,
    "output_size": 10
}

# 2. Data Loading (Raw 0-255 values)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0), 
    transforms.Lambda(lambda x: torch.flatten(x))
])

# Ensure shuffle=False to keep comparison deterministic
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=False)

# 3. Model Definition
class ComparisonNN(nn.Module):
    def __init__(self):
        super(ComparisonNN, self).__init__()
        self.layer1 = nn.Linear(params["input_size"], params["size_of_hidden_layer"])
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(params["size_of_hidden_layer"],params["size_of_hidden_layer"])
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(params["size_of_hidden_layer"], params["output_size"])
        self.relu3 = nn.ReLU() 

        # Initialization: All weights and biases = 1.0
        with torch.no_grad():
            self.layer1.weight.fill_(1.0)
            self.layer1.bias.fill_(1.0)
            self.layer2.weight.fill_(1.0)
            self.layer2.bias.fill_(1.0)            
            self.layer3.weight.fill_(1.0)
            self.layer3.bias.fill_(1.0)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        return x

model = ComparisonNN()

# 4. Optimizer and Loss
# reduction='mean' is the PyTorch default (divides sum by N * Output_Size)
criterion = nn.MSELoss() 
optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])

# 5. Training (1 Batch, 1 Epoch)
model.train()
data, target = next(iter(train_loader)) # Get the full batch of 60,000

# One-hot encode targets
target_one_hot = torch.nn.functional.one_hot(target, num_classes=10).float()

optimizer.zero_grad()
output = model(data)
loss = 0.5 * torch.sum((output - target_one_hot)**2) / params["batch_size"]
loss.backward()
print("--- GRADIENTS (Check these first) ---")
print("Layer 3 Weight Grad (First 5):")
print(model.layer3.weight.grad[0][:5]) 
print("Layer 2 Weight Grad (First 5):")
print(model.layer2.weight.grad[0][:5]) 
print("Layer 1 Weight Grad (First 5):")
print(model.layer1.weight.grad[0][:5])
optimizer.step()

# 6. Printing the Whole Matrices
print("="*50)
print("FINAL WEIGHTS: LAYER 1 (Hidden Layer)")
print(f"Shape: {model.layer1.weight.shape}")
print(model.layer1.weight.detach())

print("\n" + "="*50)
print("FINAL BIAS: LAYER 1")
print(model.layer1.bias.detach())

print("\n" + "="*50)
print("FINAL WEIGHTS: LAYER 2 (Output Layer)")
print(f"Shape: {model.layer2.weight.shape}")
print(model.layer2.weight.detach())

print("\n" + "="*50)
print("FINAL BIAS: LAYER 2")
print(model.layer2.bias.detach())

print("\n" + "="*50)
print("FINAL WEIGHTS: LAYER 3 (Output Layer)")
print(f"Shape: {model.layer3.weight.shape}")
print(model.layer3.weight.detach())

print("\n" + "="*50)
print("FINAL BIAS: LAYER 2")
print(model.layer3.bias.detach())