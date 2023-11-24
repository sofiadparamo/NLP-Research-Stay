import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate some toy data for the neural network
# We'll use a simple XOR problem for demonstration
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define the neural network architecture
class XORNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(XORNeuralNetwork, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden_output = torch.sigmoid(self.hidden_layer(x))
        output = torch.sigmoid(self.output_layer(hidden_output))
        return output

# Initialize the neural network
input_dim = 2
hidden_dim = 2
output_dim = 1
model = XORNeuralNetwork(input_dim, hidden_dim, output_dim)

# Define the loss function (Mean Squared Error) and the optimizer (Stochastic Gradient Descent)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 16000

for epoch in range(epochs):
    optimizer.zero_grad()
    # Forward pass
    predictions = model(X)
    # Calculate the loss
    loss = criterion(predictions, y)
    # Backpropagation
    loss.backward()
    # Update weights
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Testing the trained neural network
test_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
with torch.no_grad():
    predictions = model(test_data)

print("\nPredictions:")
print(predictions)