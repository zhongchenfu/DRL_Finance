"""
R-PPO network model
- Low frequency (weekly): 2-layer stacked LSTM
- Medium frequency (daily): 2-layer stacked LSTM
- High frequency (5-minute): 3-layer dilated LSTM (dilation rates 1, 2, 4)
- Fused features feed into PPO actor/critic networks

Registered as a Stable Baselines 3 BaseFeaturesExtractor.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import numpy as np


class DilatedLSTMCell(nn.Module):
    """Single-layer dilated LSTM module that consumes every dilation-th sequence step."""
    def __init__(self, input_size: int, hidden_size: int, dilation: int = 1):
        super().__init__()
        self.dilation = dilation
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        Extract subsequences by dilation, run them through the LSTM, and concatenate outputs.
        """
        B, T, F = x.shape
        d = self.dilation
        # Group T by dilation.
        # pad T to be divisible by d
        pad = (d - T % d) % d
        if pad > 0:
            x = torch.cat([x, x[:, -pad:, :]], dim=1)  # Pad at the end.
        T2 = x.shape[1]
        # reshape: (B, d, T2//d, F) then (B*d, T2//d, F)
        x_r = x.reshape(B, T2 // d, d, F).permute(0, 2, 1, 3).reshape(B * d, T2 // d, F)
        out, _ = self.lstm(x_r)  # (B*d, T2//d, hidden)
        # back to (B, d, T2//d, hidden) -> (B, T2, hidden)
        H = out.shape[-1]
        out = out.reshape(B, d, T2 // d, H).permute(0, 2, 1, 3).reshape(B, T2, H)
        # trim padding
        out = out[:, :T, :]
        return out


class DilatedLSTMStack(nn.Module):
    """3-layer stacked Dilated LSTM (dilation rates 1, 2, 4) for high-frequency data."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell1 = DilatedLSTMCell(input_size, hidden_size, dilation=1)
        self.cell2 = DilatedLSTMCell(hidden_size, hidden_size, dilation=2)
        self.cell3 = DilatedLSTMCell(hidden_size, hidden_size, dilation=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.cell1(x)
        h2 = self.cell2(h1)
        h3 = self.cell3(h2)
        return h3[:, -1, :]  # Use the final time step.


class StackedLSTM(nn.Module):
    """2-layer stacked standard LSTM for low-/medium-frequency data."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out[:, -1, :]  # Use the final time step.


class MultiFreqFeaturesExtractor(BaseFeaturesExtractor):
    """
    Multi-timeframe fusion feature extractor (SB3 BaseFeaturesExtractor).

    Observation-space format (observation_space):
      Box(shape = (low_len * n_low_feat + mid_len * n_mid_feat
                   + high_len * n_high_feat + portfolio_features,))
      The three flat market vectors are concatenated with the portfolio state.

    Pass via policy_kwargs:
      features_extractor_kwargs = dict(
          low_len=8, low_features=n,
          mid_len=15, mid_features=n,
          high_len=48, high_features=n,
          hidden_size=64,
      )
    """
    def __init__(self, observation_space: gym.Space,
                 low_len: int = 8,   low_features: int = 10,
                 mid_len: int = 15,  mid_features: int = 15,
                 high_len: int = 48, high_features: int = 6,
                 hidden_size: int = 32,
                 portfolio_features: int = 3):
        features_dim = hidden_size * 3 + portfolio_features
        super().__init__(observation_space, features_dim)

        self.low_len   = low_len
        self.mid_len   = mid_len
        self.high_len  = high_len
        self.low_feat  = low_features
        self.mid_feat  = mid_features
        self.high_feat = high_features
        self.hidden    = hidden_size
        self.portfolio_features = portfolio_features

        # Low frequency: 2-layer Stacked LSTM
        self.low_net  = StackedLSTM(low_features,  hidden_size)
        # Medium frequency: 2-layer Stacked LSTM
        self.mid_net  = StackedLSTM(mid_features,  hidden_size)
        # High frequency: 3-layer Dilated LSTM
        self.high_net = DilatedLSTMStack(high_features, hidden_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: (batch, total_flat)
        """
        B = observations.shape[0]
        # Flatten any extraneous dummy dimensions added by SB3 (like [128, 1, 1758])
        observations = observations.reshape(B, -1)

        # Split the three market segments.
        low_size  = self.low_len  * self.low_feat
        mid_size  = self.mid_len  * self.mid_feat
        high_size = self.high_len * self.high_feat

        low_flat  = observations[:, :low_size]
        mid_flat  = observations[:, low_size:low_size + mid_size]
        high_flat = observations[:, low_size + mid_size: low_size + mid_size + high_size]
        portfolio_state = observations[
            :, low_size + mid_size + high_size: low_size + mid_size + high_size + self.portfolio_features
        ]

        # reshape to (batch, seq, features)
        low_seq  = low_flat.reshape(B, self.low_len,  self.low_feat)
        mid_seq  = mid_flat.reshape(B, self.mid_len,  self.mid_feat)
        high_seq = high_flat.reshape(B, self.high_len, self.high_feat)

        h_low  = self.low_net(low_seq)    # (B, hidden)
        h_mid  = self.mid_net(mid_seq)    # (B, hidden)
        h_high = self.high_net(high_seq)  # (B, hidden)

        fused = torch.cat([h_low, h_mid, h_high, portfolio_state], dim=-1)
        return fused


if __name__ == "__main__":
   
    import gymnasium as gym

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    low_len, low_feat   = 8,   10
    mid_len, mid_feat   = 15,  15
    high_len, high_feat = 48,  6  # Shorten the high-frequency length for easier testing.
    hidden = 32

    portfolio_features = 3
    total = low_len * low_feat + mid_len * mid_feat + high_len * high_feat + portfolio_features
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total,), dtype=np.float32)

    net = MultiFreqFeaturesExtractor(
        obs_space,
        low_len=low_len,   low_features=low_feat,
        mid_len=mid_len,   mid_features=mid_feat,
        high_len=high_len, high_features=high_feat,
        hidden_size=hidden,
        portfolio_features=portfolio_features,
    ).to(device)  # Move the model to the target device.

    dummy = torch.randn(4, total).to(device)  # batch_size=4; move to the target device.
    out = net(dummy)
    print(f"Output shape: {out.shape}")  # expect (4, 99)
    assert out.shape == (4, hidden * 3 + portfolio_features)
  
