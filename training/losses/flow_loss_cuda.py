import torch

def compute_flow_matching_loss(flow_matcher, z_target: torch.Tensor, h_context: torch.Tensor, mask: torch.Tensor = None):
    """
    最优传输流匹配损失 (Rectified Flow Matching Loss) 的 CUDA/PyTorch 版本。
    这是整个系统唯一且终极的因果评判标准。
    
    Args:
        flow_matcher: 实例化的 FlowMatcher 喷嘴网络
        z_target: (B, L, d_model) 真实世界里下一句话的算子坐标 (来自 Phase 0.5)
        h_context: (B, L, d_model) Mamba 根据历史计算出的当前势能状态
        mask: (B, L) bool 类型的序列掩码，过滤掉 Padding
    """
    B, L, D = z_target.shape
    device = z_target.device
    dtype = z_target.dtype
    
    t = torch.rand(size=(B, L, 1), device=device, dtype=dtype)
    x_0 = torch.randn(size=(B, L, D), device=device, dtype=dtype)
    
    # 物理积分公式
    x_t = (1.0 - t) * x_0 + t * z_target
    v_target = z_target - x_0
    
    # 网络预测
    v_pred = flow_matcher(x_t, t, h_context)
    
    # 计算逐点动能误差 (B, L, D) -> (B, L)
    squared_error = torch.mean(torch.square(v_pred - v_target), dim=-1)
    
    # 遮蔽 Padding 部分的误差 (极度安全写法，防止 NaN * 0 = NaN 导致全盘崩溃)
    if mask is not None:
        squared_error = torch.where(mask == 0, torch.zeros_like(squared_error), squared_error)
        loss = torch.sum(squared_error) / torch.clamp(torch.sum(mask), min=1.0)
    else:
        loss = torch.mean(squared_error)
        
    return loss
