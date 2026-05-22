import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


from model.mamba_planner import MambaPlanner
from model.mamba_mlx.mamba_mlx import MambaConfig
from model.flow_matcher import FlowMatcher
from training.core.dataloader import TextDocumentDataLoader
from training.core.char_tokenizer import CharTokenizer
from training.core.checkpoint import Checkpointer
from training.core.schedule import linear_warmup_schedule
from training.losses.flow_loss import compute_flow_matching_loss
from training.core.args import get_training_parser

# Dynamic Import of the Phase 0.5 Distilled Model
from distilled_emb.model import TinyCharEncoder

class E2EModel(nn.Module):
    """
    联合训练容器：将 Mamba (大脑) 和 FlowMatcher (喷嘴) 缝合，
    方便 MLX 对这两者一并进行梯度反向传播。
    """
    def __init__(self, mamba, flow):
        super().__init__()
        self.mamba = mamba
        self.flow = flow
        
    def __call__(self, main_stream, z_target, mask):
        # 1. 大脑思考：Mamba 吸收历史算子，积分出当前的纯净势能 h_context
        h_context = self.mamba(main_stream, aux_streams=[])
        
        # 2. 嘴巴吹风：FlowMatcher 在 h_context 的势能场下，尝试吹出微观轨迹
        loss = compute_flow_matching_loss(self.flow, z_target, h_context, mask=mask)
        return loss

def main():
    parser = get_training_parser("Phase 1: Mamba & Flow Matching Training (On-the-fly TinyBERT)")
    parser.add_argument("--max_episode_len", type=int, default=None, help="Sequence chunking limit.")
    parser.add_argument("--data_path", type=str, default="data/Basic_ZH/chunked_mixed_omni.parquet", help="Path to the parquet training data.")
    parser.add_argument("--tinybert_ckpt", type=str, default="checkpoints/distilled/tinybert_pt_v1_step_100000", help="Path to frozen Phase 0.5 distilled TinyBERT.")
    parser.add_argument("--d_model", type=int, default=1024, help="Dimension of the Mamba and FlowMatcher physical backbone.")
    parser.add_argument("--mamba_d_state", type=int, default=16, help="Mamba internal state dimension.")
    parser.add_argument("--mamba_d_conv", type=int, default=4, help="Mamba internal conv dimension.")
    parser.add_argument("--mamba_expand", type=int, default=2, help="Mamba internal expansion factor.")
    parser.add_argument("--flow_hidden_dim", type=int, default=2048, help="Hidden expansion dimension for the Flow Matcher nozzle.")
    args = parser.parse_args()

    # d_model 现在是一个可以被命令行控制的参数
    d_model = args.d_model

    # 1. 实例化真正的基础词表与 DataLoader
    tokenizer = CharTokenizer()
    dataloader = TextDocumentDataLoader(
        parquet_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_episode_len=args.max_episode_len
    )

    # 2. 嗅探并实例化 Phase 0.5 (自动读取维度)
    print(f"Loading Frozen Phase 0.5 TinyCharEncoder from {args.tinybert_ckpt}...")
    
    safetensors_path = args.tinybert_ckpt
    if not safetensors_path.endswith(".safetensors"):
        safetensors_path = f"{args.tinybert_ckpt}/model.safetensors"
        if not os.path.exists(safetensors_path):
            safetensors_path = f"{args.tinybert_ckpt}/tinybert.safetensors"
            if not os.path.exists(safetensors_path):
                raise FileNotFoundError(f"Could not find .safetensors in {args.tinybert_ckpt}. MLX requires .safetensors format (not PyTorch .pt)!")

    # 在实例化前，先暴力嗅探 safetensors 里的输出层维度 (z_dim)
    weights = mx.load(safetensors_path)
    try:
        sniffed_z_dim = weights['out_proj.weight'].shape[0]
        print(f"[Auto-Sniff] Successfully sniffed Z_dim = {sniffed_z_dim} from checkpoint!")
    except KeyError:
        raise ValueError("Cannot find 'out_proj.weight' in the checkpoint. Is this a valid TinyCharEncoder?")

    # 使用嗅探到的维度初始化
    tiny_encoder = TinyCharEncoder(vocab_size=tokenizer.vocab_size, z_dim=sniffed_z_dim)
    tiny_encoder.load_weights(list(weights.items()))
    tiny_encoder.freeze()
    print("Phase 0.5 TinyCharEncoder loaded and frozen. We are now self-bootstrapping!")
    
    # 3. 实例化真正的训练目标：Mamba大脑 + 喷嘴
    mamba_cfg = MambaConfig(
        d_model=d_model, 
        n_layers=2,
        d_state=args.mamba_d_state,
        d_conv=args.mamba_d_conv,
        expand_factor=args.mamba_expand
    )
    mamba_planner = MambaPlanner(mamba_cfg)
    flow_matcher = FlowMatcher(d_model=d_model, hidden_dim=args.flow_hidden_dim)
    
    e2e_model = E2EModel(mamba_planner, flow_matcher)
    mx.eval(e2e_model.parameters())

    optimizer = optim.AdamW(learning_rate=args.lr)

    # 4. Checkpointer
    checkpointer = Checkpointer(args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("mamba_planner", mamba_planner)
    checkpointer.register_model("flow_matcher", flow_matcher)
    checkpointer.register_dataloader("dataloader_phase1", dataloader)
    checkpointer.register_optimizer("optimizer", optimizer)
    checkpointer.register_args(args)
    
    start_step = checkpointer.load(args.resume_from) if args.resume_from else (checkpointer.load_latest() if args.auto_resume else 0)

    # 5. 定义 MLX 梯度磁带
    def loss_fn(model, main_stream, z_target, mask):
        return model(main_stream, z_target, mask)
        
    step_fn = nn.value_and_grad(e2e_model, loss_fn)
    
    @mx.compile
    def train_step(main_stream, z_target, masks):
        loss, grads = step_fn(e2e_model, main_stream, z_target, masks)
        clipped_grads, global_norm = optim.clip_grad_norm(grads, 1.0)
        return loss, clipped_grads, global_norm

    print(f"Starting Phase 1 E2E Physics Training. Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    global_step = start_step

    try:
        for epoch in range(dataloader.current_epoch, args.epochs):
            for ids_t, att_t, sen_t in dataloader:
                global_step += 1
                optimizer.learning_rate = linear_warmup_schedule(global_step, args.lr, args.warmup_steps)
                
                B, S, T = ids_t.shape
                if S < 2: 
                    continue # 至少两句话才能构成“上一句预测下一句”
                
                # A. 展平批次和句子维度，一次性送入 TinyCharEncoder (秒算全部 Z)
                flat_ids = ids_t.reshape(-1, T)
                flat_att = att_t.reshape(-1, T)
                
                # B. 计算 Z_target 并重新折叠回 3D 文档结构
                z_flat = tiny_encoder(flat_ids, attention_mask=flat_att)
                z_truth = z_flat.reshape(B, S, -1) # 形如 (Batch, Max_Sentences, Z_dim)
                
                # 终极防御：完全 Padding 的句子在 TinyCharEncoder 里可能会除以 1e-8 产生极大值或 NaN，
                # 我们这里用 sentence_mask (sen_t) 强行归零，彻底切断物理空间里的 NaN 污染源。
                z_truth = mx.where(sen_t[:, :, None] == 0, mx.zeros_like(z_truth), z_truth)
                
                # C. 时空错位法 (Past -> Future)
                f_t_input = z_truth[:, :-1, :]
                z_target_truth = z_truth[:, 1:, :]
                masks_shifted = sen_t[:, 1:]
                
                # D. 物理瞬时解算与反向传播
                loss, grads, global_norm = train_step(f_t_input, z_target_truth, masks_shifted)
                
                if mx.isnan(loss).item() or mx.isinf(loss).item() or mx.isnan(global_norm).item() or mx.isinf(global_norm).item():
                    print(f"[Anomaly] NaN/Inf Loss or Gradient detected at Step {global_step}! Skipping update.")
                    continue
                    
                optimizer.update(e2e_model, grads)
                mx.eval(e2e_model.parameters(), optimizer.state, loss)
                
                if global_step % 10 == 0:
                    print(f"Epoch {epoch+1} | Step {global_step} | Flow Match Loss (Kinetic Error): {loss.item():.4f}")
                    
                if global_step % args.save_steps == 0:
                    checkpointer.save(global_step)
                    
    except KeyboardInterrupt:
        checkpointer.save(global_step, is_emergency=True)

if __name__ == "__main__":
    main()
