import torch
import torch.nn as nn
import numpy as np 

class AttentionModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super(AttentionModel, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)

        # Attention weights
        e = self.attention(lstm_out).squeeze(-1)           # (batch_size, seq_len)
        alpha = torch.softmax(e, dim=1).unsqueeze(-1)      # (batch_size, seq_len, 1)

        # Context vector
        context = torch.sum(alpha * lstm_out, dim=1)       # (batch_size, hidden_dim * 2)

        # Final classification
        out = self.classifier(context)                     # (batch_size, 1)
        return out, alpha.squeeze(-1)                      # also return attention weights
    


    def train_model(self, data, device='cpu', num_epochs=50, patience=5, lr=1e-3):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            self.train()
            train_losses = []

            for batch_X, batch_Y in data.train_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                optimizer.zero_grad()
                outputs, _ = self(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Evaluate on validation set
            self.eval()
            val_losses = []
            with torch.no_grad():
                for val_X, val_Y in data.val_loader:
                    val_X, val_Y = val_X.to(device), val_Y.to(device)
                    val_outputs, _ = self(val_X)
                    val_loss = criterion(val_outputs, val_Y)
                    val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)

            print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        # Load best model
        self.load_state_dict(torch.load('best_model.pth'))
        print("Best model loaded.")
