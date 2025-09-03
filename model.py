import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    input_channels: int = 12        # ECG leads
    seq_length: int = 2560          # time samples per ECG
    hidden_dim: int = 64
    mlp_dim: int = 128
    num_classes: int = 6
    window_size: int = 64
    stride: int = 32
    padding: int = 16
    kernel_size: int = 5
    downsample: int = 2
    dropout_rate: float = 0.1


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 kernel_size=3, dropout_rate=0.1):
        super(ResBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.skip_conv = None
        if stride != 1 or in_channels != out_channels:
            self.skip_conv = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=1, stride=stride,
                bias=False
            )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        identity = x

        out = self.conv1(x)   # (B, out_channels, L/stride)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out) # (B, out_channels, L/stride)
        out = self.bn2(out)
        out = self.dropout2(out)

        if self.skip_conv is not None:
            identity = self.skip_conv(identity)

        out += identity
        out = self.relu(out)
        return out


class DeepConvFrontend(nn.Module):
    def __init__(self, input_channels=12, hidden_dim=64, dropout_rate=0.1):
        super(DeepConvFrontend, self).__init__()

        self.conv_in = nn.Conv1d(
            input_channels, hidden_dim,
            kernel_size=7, stride=2,
            padding=3, bias=False
        )
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(dropout_rate)

        self.layer1 = ResBlock1D(hidden_dim, hidden_dim,
                                 stride=1, kernel_size=7,
                                 dropout_rate=dropout_rate)
        self.layer2 = ResBlock1D(hidden_dim, hidden_dim,
                                 stride=2, kernel_size=7,
                                 dropout_rate=dropout_rate)
        self.layer3 = ResBlock1D(hidden_dim, hidden_dim,
                                 stride=1, kernel_size=5,
                                 dropout_rate=dropout_rate)
        self.layer4 = ResBlock1D(hidden_dim, hidden_dim,
                                 stride=2, kernel_size=5,
                                 dropout_rate=dropout_rate)

    def forward(self, x):
        # x: (B, 12, 2560)
        out = self.conv_in(x)   # (B, hidden_dim, 1280)
        out = self.bn_in(out)
        out = self.relu(out)
        out = self.dropout_in(out)

        out = self.layer1(out)  # (B, hidden_dim, 1280)
        out = self.layer2(out)  # (B, hidden_dim, 640)
        out = self.layer3(out)  # (B, hidden_dim, 640)
        out = self.layer4(out)  # (B, hidden_dim, 320)
        return out


class WindowedAttention(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=10,
                 stride=2,
                 num_heads=8,
                 conv_kernel_size=3,
                 conv_stride=1,
                 conv_padding=1):
        super(WindowedAttention, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.norm = nn.LayerNorm(dim)

        self.proj_Q = nn.Conv1d(dim, dim, kernel_size=conv_kernel_size,
                                stride=conv_stride, padding=conv_padding, bias=False)
        self.proj_K = nn.Conv1d(dim, dim, kernel_size=conv_kernel_size,
                                stride=conv_stride, padding=conv_padding, bias=False)
        self.proj_V = nn.Conv1d(dim, dim, kernel_size=conv_kernel_size,
                                stride=conv_stride, padding=conv_padding, bias=False)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape

        x = self.norm(x)

        x_t = x.permute(0, 2, 1)  # (B, D, N)
        Q_t = self.proj_Q(x_t)
        K_t = self.proj_K(x_t)
        V_t = self.proj_V(x_t)

        Q_t = F.avg_pool1d(Q_t, kernel_size=2, stride=2)  # (B, D, N//2)

        M = N // 2
        Q = Q_t.permute(0, 2, 1)  # (B, M, D)
        K = K_t.permute(0, 2, 1)  # (B, N, D)
        V = V_t.permute(0, 2, 1)  # (B, N, D)

        Q = Q.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, M, head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # (B, heads, M, head_dim)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, M, D)  # (B, M, D)

        return attn_out


class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, window_size, stride):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowedAttention(dim,
                                      kernel_size=window_size,
                                      stride=stride,
                                      num_heads=8,
                                      conv_kernel_size=3,
                                      conv_stride=1,
                                      conv_padding=1)

        self.res_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.res_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        # x: (B, seq_len, dim)
        x_norm = self.norm1(x)
        y = self.attn(x_norm)  # (B, seq_len//2, dim)

        x_res = x_norm.permute(0, 2, 1)   # (B, dim, seq_len)
        x_res = self.res_pool(x_res)      # (B, dim, seq_len//2)
        x_res = self.res_conv(x_res)      # (B, dim, seq_len//2)
        x_res = x_res.permute(0, 2, 1)    # (B, seq_len//2, dim)

        y = y + x_res
        y = y + self.mlp(self.norm2(y))   # (B, seq_len//2, dim)
        return y


class ConvTransformer(nn.Module):
    def __init__(self, input_channels=12, seq_length=2560, hidden_dim=64, mlp_dim=128, padding=0,
                 num_classes=6, num_layers=0, window_size=10, stride=2, dropout_rate=0.1):
        super(ConvTransformer, self).__init__()

        self.frontend = DeepConvFrontend(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

        num_layers = int(math.log2(320 / 10))

        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(hidden_dim, mlp_dim * (2 ** i), window_size, stride)
              for i in range(num_layers)]
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, 12, 2560)
        x = self.frontend(x)         # (B, hidden_dim, 320)
        x = x.permute(0, 2, 1)       # (B, 320, hidden_dim)

        x = self.transformer_layers(x)  # (B, 10, hidden_dim)

        out = self.mlp(x.mean(dim=1))   # (B, num_classes)
        return out
