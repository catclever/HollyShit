import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    """
    连续时间 t (0.0 ~ 1.0) 的高频位置编码。
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, t: torch.Tensor):
        half_dim = self.d_model // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -embeddings)
        embeddings = t * embeddings
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class AdaLN(nn.Module):
    """
    自适应层归一化 (Adaptive Layer Norm)。
    根据外界注入的条件 (cond)，动态计算尺度 (Scale) 和偏移量 (Shift)。
    强制性扭曲主干特征空间。
    """
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        # 输出 2 倍维度，一半用于 scale，一半用于 shift
        self.cond_proj = nn.Linear(cond_dim, d_model * 2)
        
        # 巧妙初始化：让网络初始时表现得像普通的 RMSNorm (Scale=0, Shift=0)
        nn.init.zeros_(self.cond_proj.weight)
        if self.cond_proj.bias is not None:
            nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        h = self.norm(x)
        scale_shift = self.cond_proj(cond)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        # 注意: 真正的缩放系数是 1.0 + scale，所以 scale=0 时原样输出
        return h * (1.0 + scale) + shift

class FlowResBlock(nn.Module):
    """
    带有 AdaLN 调制的残差块。
    这在 DiT (Diffusion Transformer) 中是标准设计，能有效克服特征拼接带来的坍缩问题。
    """
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.adaln1 = AdaLN(hidden_dim, cond_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.adaln2 = AdaLN(hidden_dim, cond_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # 第一次物理调制与升维
        h = self.adaln1(x, cond)
        h = self.silu(self.lin1(h))
        # 第二次物理调制与降维
        h = self.adaln2(h, cond)
        h = self.lin2(h)
        return x + h

class FlowMatcher(nn.Module):
    """
    微观动力学喷嘴 (The Motor Cortex)
    
    采用强力 AdaLN 架构，接收 Mamba 提供的绝对势能场 h_context，
    将纯高斯噪声 x_0 沿着速度场一点点“吹”向真实的目标算子 Z_target。
    """
    def __init__(self, d_model: int, hidden_dim: int = 2048):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        
        # 物理主干流的入口（从 d_model 升维到宽体 hidden_dim）
        self.in_proj = nn.Linear(d_model, hidden_dim)
        
        # 控制流的维度 = 时间嵌入维度 (d_model) + Mamba背景势能维度 (d_model)
        cond_dim = d_model * 2
        
        self.res_block1 = FlowResBlock(hidden_dim, cond_dim)
        self.res_block2 = FlowResBlock(hidden_dim, cond_dim)
        
        self.out_norm = AdaLN(hidden_dim, cond_dim)
        self.out_proj = nn.Linear(hidden_dim, d_model)
        
        # 初始化最后一层为全零，这是 Flow Matching 避免起手就梯度爆炸的标准操作！
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, h_context: torch.Tensor):
        """
        前向预测瞬时速度场。
        
        Args:
            x_t: (B, L, d_model) 当前在微观相空间中的坐标 (含有噪声)
            t: (B, L, 1) 或 (B, 1, 1) 积分时间 0.0 ~ 1.0
            h_context: (B, L, d_model) Mamba 提供的不言的宏观背景势能
            
        Returns:
            v_pred: (B, L, d_model) 喷嘴预测的瞬时速度向量
        """
        t_emb = self.time_embed(t)
        
        # 1. 组装并合成为“控制台指令” (Control Stream)
        cond = torch.cat([t_emb, h_context], dim=-1)
        
        # 2. 只有主干物质 x_t 进入骨架网络 (Material Stream)
        h = self.in_proj(x_t)
        
        # 3. 控制流强制调制物质流
        h = self.res_block1(h, cond)
        h = self.res_block2(h, cond)
        
        # 4. 最终收束并输出速度场
        h = self.out_norm(h, cond)
        v_pred = self.out_proj(h)
        return v_pred
