import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    print("[Warning] 'mamba_ssm' not installed. Please run: pip install mamba-ssm causal-conv1d")
    # Provide a dummy or error throwing class if not available during init
    class Mamba(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("mamba_ssm must be installed to use MambaPlanner_CUDA.")

class OrthogonalFieldSuperposition(nn.Module):
    """
    基于物理场叠加原理的多流融合 (Option A)。
    不使用任何 Dense 层或 Norm 层，保证信息的高维正交性与 0 污染。
    彻底保留半径 (Radius) 作为能量/确信度的物理学特性。
    """
    def __init__(self, d_model: int):
        super().__init__()
        pass
        
    def forward(self, main_stream: torch.Tensor, aux_streams: list[torch.Tensor]):
        if not aux_streams:
            return main_stream
            
        # 1. 辅助场叠加 (无论有几个外部知识流，都在这里汇聚成一个合力场)
        # 广播机制原生支持“部分流推进”：
        # - 如果是静态场：形状为 (B, 1, d_model)，自动广播到所有时间步。
        # - 如果是推进场：形状为 (B, L, d_model)，与主流在时间轴上同步推进。
        combined_aux_field = aux_streams[0]
        for aux in aux_streams[1:]:
            combined_aux_field = combined_aux_field + aux
        
        # 2. 高维空间的直接叠加 (天然的正交注射与能量叠加)
        return main_stream + combined_aux_field


class MambaPlanner(nn.Module):
    """
    The Contextual Brain (Phase 1).
    
    定位：纯上下文大脑，长程状态机与多路融合枢纽。
    不再负责直接的微观轨迹预测 (移交给了 Phase 2 的 Flow Matching)。
    唯一职责：接收主序列和多条认知流，将其淬炼压缩为一个极度纯粹、稳定的当前状态势能 h_context。
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        
        # 1. 纯物理叠加的融合场
        self.stream_fuser = OrthogonalFieldSuperposition(self.d_model)
        
        # 2. 纯粹的状态机引擎 (使用官方的 mamba_ssm 库，底层自带 Triton 加速)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
    def forward(self, main_stream: torch.Tensor, aux_streams: list[torch.Tensor] = None):
        """
        Forward pass for Contextual Brain.
        
        Args:
            main_stream: (B, L, d_model) - 主方向特征流 (静态视网膜的单字序列)
            aux_streams: List of (B, *, d_model) - 并行的辅助场 (如全局 RAG 向量)
            
        Returns:
            h_context: (B, L, d_model) - 绝对条件控制向量 (为 downstream Phase 2 提供场条件)
        """
        if aux_streams is None:
            aux_streams = []
            
        # Step 1: 物理场的叠加与注射
        fused_input = self.stream_fuser(main_stream, aux_streams)
        
        # Step 2: 状态机的长程积分 (淬炼出唯一的当前势能)
        h_context = self.mamba(fused_input)
        
        return h_context
