import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def create_sequences(input_data, output_data, seq_length):
    X, y = [], []
    for i in range(len(input_data) - seq_length):
        X.append(input_data[i:i+seq_length])
        y.append(output_data[i+seq_length])  
    return np.array(X), np.array(y)

seq_length = 10 
X, y = create_sequences(input_signal, response, seq_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # (batch, seq_length, features)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  
        out = self.fc(lstm_out[:, -1, :])  
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train, y_train = X_train.to(device), y_train.to(device)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

model.eval()
X_test = X_test.to(device)
y_test = y_test.to(device)
y_pred_test = model(X_test).detach().cpu().numpy()

plt.figure(figsize=(8, 5))
plt.plot(time[seq_length:len(y_pred_test)+seq_length], y_test.cpu().numpy(), label="True Response")
plt.plot(time[seq_length:len(y_pred_test)+seq_length], y_pred_test, label="LSTM Prediction", linestyle="dashed")
plt.xlabel("Time (s)")
plt.ylabel("Response")
plt.legend()
plt.grid()
plt.show()

