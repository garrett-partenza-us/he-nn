import torch
import torch.nn as nn


# ShallowNet class
class ShallowNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ShallowNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Init neural network
input_size = 16
hidden_size = 8
output_size = 1
model = ShallowNet(input_size, hidden_size, output_size)

print(model)

# Move the model to the MPS device if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Create a sample input tensor
sample_input = torch.randn(1, input_size).to(device)

# Perform a forward pass
output = model(sample_input)
print(output)
