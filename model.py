import torch
import torch.nn as nn

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
