"""
R-PPO 网络模型
- 低频 (周线): 2层 Stacked LSTM
- 中频 (日线): 2层 Stacked LSTM
- 高频 (5分钟线): 3层 Dilated LSTM (扩张率1,2,4)
- 融合后接PPO Actor/Critic网络

注册为 Stable Baselines 3 的 BaseFeaturesExtractor
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import numpy as np


class DilatedLSTMCell(nn.Module):
    """单层带扩张率的LSTM模块（在序列方向上每隔 dilation 步接受一次输入）"""
    def __init__(self, input_size: int, hidden_size: int, dilation: int = 1):
        super().__init__()
        self.dilation = dilation
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        通过扩张抽取子序列，分别过LSTM，拼接输出
        """
        B, T, F = x.shape
        d = self.dilation
        # 将T按照dilation分组
        # pad T to be divisible by d
        pad = (d - T % d) % d
        if pad > 0:
            x = torch.cat([x, x[:, -pad:, :]], dim=1)  # 末尾填充
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
    """3层堆叠Dilated LSTM (扩张率 1, 2, 4)，用于高频数据"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell1 = DilatedLSTMCell(input_size, hidden_size, dilation=1)
        self.cell2 = DilatedLSTMCell(hidden_size, hidden_size, dilation=2)
        self.cell3 = DilatedLSTMCell(hidden_size, hidden_size, dilation=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.cell1(x)
        h2 = self.cell2(h1)
        h3 = self.cell3(h2)
        return h3[:, -1, :]  # 取最后时间步


class StackedLSTM(nn.Module):
    """2层堆叠标准LSTM，用于低频/中频数据"""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out[:, -1, :]  # 取最后时间步


class MultiFreqFeaturesExtractor(BaseFeaturesExtractor):
    """
    多时间维度融合特征提取器 (SB3 BaseFeaturesExtractor)

    观测空间格式 (observation_space):
      Box(shape = (low_len * n_low_feat + mid_len * n_mid_feat + high_len * n_high_feat,))
      即三段flat向量拼接

    需通过 policy_kwargs 传入:
      features_extractor_kwargs = dict(
          low_len=8, low_features=n,
          mid_len=15, mid_features=n,
          high_len=240, high_features=n,
          hidden_size=64,
      )
    """
    def __init__(self, observation_space: gym.Space,
                 low_len: int = 8,   low_features: int = 10,
                 mid_len: int = 15,  mid_features: int = 15,
                 high_len: int = 240, high_features: int = 6,
                 hidden_size: int = 64):
        features_dim = hidden_size * 3  # 三路融合后维度
        super().__init__(observation_space, features_dim)

        self.low_len   = low_len
        self.mid_len   = mid_len
        self.high_len  = high_len
        self.low_feat  = low_features
        self.mid_feat  = mid_features
        self.high_feat = high_features
        self.hidden    = hidden_size

        # 低频: 2层 Stacked LSTM
        self.low_net  = StackedLSTM(low_features,  hidden_size)
        # 中频: 2层 Stacked LSTM
        self.mid_net  = StackedLSTM(mid_features,  hidden_size)
        # 高频: 3层 Dilated LSTM
        self.high_net = DilatedLSTMStack(high_features, hidden_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: (batch, total_flat)
        """
        B = observations.shape[0]
        # Flatten any extraneous dummy dimensions added by SB3 (like [128, 1, 1758])
        observations = observations.reshape(B, -1)

        # 分割三段
        low_size  = self.low_len  * self.low_feat
        mid_size  = self.mid_len  * self.mid_feat
        high_size = self.high_len * self.high_feat

        low_flat  = observations[:, :low_size]
        mid_flat  = observations[:, low_size:low_size + mid_size]
        high_flat = observations[:, low_size + mid_size: low_size + mid_size + high_size]

        # reshape to (batch, seq, features)
        low_seq  = low_flat.reshape(B, self.low_len,  self.low_feat)
        mid_seq  = mid_flat.reshape(B, self.mid_len,  self.mid_feat)
        high_seq = high_flat.reshape(B, self.high_len, self.high_feat)

        h_low  = self.low_net(low_seq)    # (B, hidden)
        h_mid  = self.mid_net(mid_seq)    # (B, hidden)
        h_high = self.high_net(high_seq)  # (B, hidden)

        fused = torch.cat([h_low, h_mid, h_high], dim=-1)  # (B, hidden*3)
        return fused


if __name__ == "__main__":
   
    import gymnasium as gym

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    low_len, low_feat   = 8,   10
    mid_len, mid_feat   = 15,  15
    high_len, high_feat = 240, 6
    hidden = 32

    total = low_len * low_feat + mid_len * mid_feat + high_len * high_feat
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total,), dtype=np.float32)

    net = MultiFreqFeaturesExtractor(
        obs_space,
        low_len=low_len,   low_features=low_feat,
        mid_len=mid_len,   mid_features=mid_feat,
        high_len=high_len, high_features=high_feat,
        hidden_size=hidden,
    ).to(device)  # 将模型移至目标设备

    dummy = torch.randn(4, total).to(device)  # batch_size=4，移至目标设备
    out = net(dummy)
    print(f"Output shape: {out.shape}")  # expect (4, 96)
    assert out.shape == (4, hidden * 3)
  
