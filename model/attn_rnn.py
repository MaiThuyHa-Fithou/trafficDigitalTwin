"""
attn_rnn.py – Mô hình Attn-RNN (LSTM + Attention) bằng PyTorch.

Kiến trúc theo bài báo ICATE 2026:
  LSTM(128) → LSTM(64) → LSTM(32) → Attention → FC → output

Thay thế TensorFlow/Keras bằng PyTorch để tương thích môi trường hiện tại.
"""
import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """
    Cơ chế Attention:
      score_t = tanh(W · h_t)
      alpha_t = softmax(v · score_t)
      context = Σ alpha_t * h_t
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor):
        # lstm_out: (batch, seq, hidden)
        score   = torch.tanh(self.W(lstm_out))           # (B, T, H)
        weights = torch.softmax(self.v(score), dim=1)    # (B, T, 1)
        context = (weights * lstm_out).sum(dim=1)        # (B, H)
        return context, weights.squeeze(-1)              # weights: (B, T)


class AttnRNN(nn.Module):
    """
    LSTM nhiều lớp + Attention + FC head.
    """
    def __init__(self,
                 input_size:   int,
                 hidden_sizes: list = None,
                 dropout:      float = 0.2):
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 64, 32]

        self.lstm1 = nn.LSTM(input_size,      hidden_sizes[0],
                             batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1],
                             batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2],
                             batch_first=True)

        self.attention = AttentionLayer(hidden_sizes[2])

        self.fc = nn.Sequential(
            nn.Linear(hidden_sizes[2], 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, n_features)
        o1, _ = self.lstm1(x)
        o2, _ = self.lstm2(o1)
        o3, _ = self.lstm3(o2)
        context, attn_w = self.attention(o3)
        out = self.fc(context).squeeze(-1)      # (batch,)
        return out, attn_w


class BaselineLSTM(nn.Module):
    """LSTM cơ bản – baseline so sánh (không Attention)."""
    def __init__(self, input_size: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True,
                            num_layers=2, dropout=0.1)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1), None


def build_model(input_size: int,
                hidden_sizes: list = None,
                dropout: float = 0.2) -> AttnRNN:
    """Factory function – tương thích giao diện cũ."""
    return AttnRNN(input_size=input_size,
                   hidden_sizes=hidden_sizes,
                   dropout=dropout)
