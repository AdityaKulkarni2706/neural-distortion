import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import wavfile
import json

#CONFIG

HIDDEN_SIZE = 16  # Small and fast
EPOCHS = 500
LEARNING_RATE = 0.01
NOISE_LEVEL = 0.001

print("Loading WAV files...")
fs, input_audio = wavfile.read("audioFiles/dataset_input.wav")
fs, target_audio = wavfile.read("audioFiles/dataset_target.wav")

if input_audio.dtype == 'int16':
    input_audio = input_audio.astype(np.float32) / 32768.0
    target_audio = target_audio.astype(np.float32) / 32768.0

X = torch.tensor(input_audio).unsqueeze(1)
y = torch.tensor(target_audio).unsqueeze(1)

class DistortionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.fc2(x)


model = DistortionNet()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

print(f"Training on {len(X)} samples for {EPOCHS} epochs...")


for epoch in range(EPOCHS):
    model.train()

    noise = torch.randn_like(X) * NOISE_LEVEL
    X_noisy = X + noise
    optimizer.zero_grad()
    outputs = model(X_noisy)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")
        

print("Exporting weights to 'model_weights.json'...")
weights = {
    "layer1_weights": model.fc1.weight.detach().numpy().flatten().tolist(),
    "layer1_bias":    model.fc1.bias.detach().numpy().flatten().tolist(),
    "layer2_weights": model.fc2.weight.detach().numpy().flatten().tolist(),
    "layer2_bias":    model.fc2.bias.detach().numpy().flatten().tolist()
}

with open("Weights/model_weights.json", "w") as f:
    json.dump(weights, f, indent=4)

print("Done!")

