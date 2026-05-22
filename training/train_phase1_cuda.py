import argparse
import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mamba_planner_cuda import MambaPlanner
from model.flow_matcher_cuda import FlowMatcher
from training.core.dataloader import TextDocumentDataLoader
from training.core.char_tokenizer import CharTokenizer
from training.core.schedule import linear_warmup_schedule
from training.losses.flow_loss_cuda import compute_flow_matching_loss
from training.core.args import get_training_parser

# Dynamic Import of the Phase 0.5 Distilled Model
from distilled_emb.model_cuda import TinyCharEncoderCUDA

class E2EModel(nn.Module):
    """
    联合训练容器：将 Mamba (大脑) 和 FlowMatcher (喷嘴) 缝合
    """
    def __init__(self, mamba, flow):
        super().__init__()
        self.mamba = mamba
        self.flow = flow
        
    def forward(self, main_stream, z_target, mask):
        # 1. 大脑思考：Mamba 吸收历史算子，积分出当前的纯净势能 h_context
        h_context = self.mamba(main_stream, aux_streams=[])
        
        # 2. 嘴巴吹风：FlowMatcher 在 h_context 的势能场下，尝试吹出微观轨迹
        loss = compute_flow_matching_loss(self.flow, z_target, h_context, mask=mask)
        return loss

def load_checkpoint(out_dir, prefix, model, optimizer, dataloader, device):
    """Simple inline PyTorch checkpointer"""
    import glob
    import re
    if not os.path.exists(out_dir):
        return 0
    
    # Find latest step
    dirs = glob.glob(os.path.join(out_dir, f"{prefix}_step_*" if prefix else "step_*"))
    if not dirs:
        return 0
        
    max_step = -1
    latest_dir = None
    for d in dirs:
        m = re.search(r'step_(\d+)', d)
        if m:
            step = int(m.group(1))
            if step > max_step:
                max_step = step
                latest_dir = d
                
    if latest_dir is None:
        return 0
        
    print(f"Resuming from checkpoint {latest_dir}...")
    
    # Load Model
    if os.path.exists(os.path.join(latest_dir, "e2e_model.pt")):
        model.load_state_dict(torch.load(os.path.join(latest_dir, "e2e_model.pt"), map_location=device, weights_only=True))
    
    # Load Optimizer
    if os.path.exists(os.path.join(latest_dir, "optimizer.pt")):
        optimizer.load_state_dict(torch.load(os.path.join(latest_dir, "optimizer.pt"), map_location=device, weights_only=True))
        
    # Load Dataloader
    if os.path.exists(os.path.join(latest_dir, "dataloader_phase1.json")):
        with open(os.path.join(latest_dir, "dataloader_phase1.json"), "r") as f:
            dataloader.load_state_dict(json.load(f))
            
    return max_step

def save_checkpoint(out_dir, prefix, step, model, optimizer, dataloader):
    dir_name = f"{prefix}_step_{step}" if prefix else f"step_{step}"
    save_path = os.path.join(out_dir, dir_name)
    os.makedirs(save_path, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_path, "e2e_model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
    
    with open(os.path.join(save_path, "dataloader_phase1.json"), "w") as f:
        json.dump(dataloader.state_dict(), f)
    print(f"Saved checkpoint to {save_path}")

def main():
    parser = get_training_parser("Phase 1: Mamba & Flow Matching Training (CUDA Version)")
    parser.add_argument("--max_episode_len", type=int, default=None, help="Sequence chunking limit.")
    parser.add_argument("--data_path", type=str, default="data/Basic_ZH/chunked_mixed_omni.parquet", help="Path to the parquet training data.")
    parser.add_argument("--tinybert_ckpt", type=str, default="checkpoints/distilled/tinybert_pt_v1_step_100000", help="Path to frozen Phase 0.5 distilled TinyBERT.")
    parser.add_argument("--d_model", type=int, default=1024, help="Dimension of the Mamba and FlowMatcher physical backbone.")
    parser.add_argument("--mamba_d_state", type=int, default=16, help="Mamba internal state dimension.")
    parser.add_argument("--mamba_d_conv", type=int, default=4, help="Mamba internal conv dimension.")
    parser.add_argument("--mamba_expand", type=int, default=2, help="Mamba internal expansion factor.")
    parser.add_argument("--flow_hidden_dim", type=int, default=2048, help="Hidden expansion dimension for the Flow Matcher nozzle.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    d_model = args.d_model

    # 1. 实例化真正的基础词表与 DataLoader
    tokenizer = CharTokenizer()
    dataloader = TextDocumentDataLoader(
        parquet_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_episode_len=args.max_episode_len,
        backend='torch'
    )

    # 2. 嗅探并实例化 Phase 0.5 (自动读取维度)
    print(f"Loading Frozen Phase 0.5 TinyCharEncoder from {args.tinybert_ckpt}...")
    
    safetensors_path = args.tinybert_ckpt
    if not safetensors_path.endswith(".safetensors"):
        safetensors_path = f"{args.tinybert_ckpt}/model.safetensors"
        if not os.path.exists(safetensors_path):
            safetensors_path = f"{args.tinybert_ckpt}/tinybert.safetensors"
            if not os.path.exists(safetensors_path):
                raise FileNotFoundError(f"Could not find .safetensors in {args.tinybert_ckpt}.")

    weights = load_file(safetensors_path)
    try:
        sniffed_z_dim = weights['out_proj.weight'].shape[0]
        print(f"[Auto-Sniff] Successfully sniffed Z_dim = {sniffed_z_dim} from checkpoint!")
    except KeyError:
        raise ValueError("Cannot find 'out_proj.weight' in the checkpoint. Is this a valid TinyCharEncoder?")

    # 使用嗅探到的维度初始化
    tiny_encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size, z_dim=sniffed_z_dim)
    tiny_encoder.load_state_dict(weights, strict=False)
    for param in tiny_encoder.parameters():
        param.requires_grad = False
    tiny_encoder.eval()
    tiny_encoder.to(device)
    print("Phase 0.5 TinyCharEncoder loaded and frozen. We are now self-bootstrapping!")
    
    # 3. 实例化真正的训练目标：Mamba大脑 + 喷嘴
    mamba_planner = MambaPlanner(
        d_model=d_model, 
        d_state=args.mamba_d_state,
        d_conv=args.mamba_d_conv,
        expand=args.mamba_expand
    )
    flow_matcher = FlowMatcher(d_model=d_model, hidden_dim=args.flow_hidden_dim)
    
    e2e_model = E2EModel(mamba_planner, flow_matcher).to(device)
    
    optimizer = optim.AdamW(e2e_model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() # AMP 加速

    # 4. 恢复状态
    prefix = args.ckpt_prefix if args.ckpt_prefix else "p1_flow_cuda"
    start_step = 0
    if args.resume_from or args.auto_resume:
        load_target = args.resume_from if args.resume_from else args.out_dir
        start_step = load_checkpoint(load_target, prefix, e2e_model, optimizer, dataloader, device)

    print(f"Starting Phase 1 E2E Physics Training on CUDA. Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    global_step = start_step

    try:
        for epoch in range(dataloader.current_epoch, args.epochs):
            for ids_t, att_t, sen_t in dataloader:
                global_step += 1
                
                # Update Learning Rate
                current_lr = linear_warmup_schedule(global_step, args.lr, args.warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # Move to device
                ids_t, att_t, sen_t = ids_t.to(device), att_t.to(device), sen_t.to(device)
                
                B, S, T = ids_t.shape
                if S < 2: 
                    continue # 至少两句话才能构成“上一句预测下一句”
                
                # A. 展平批次和句子维度，一次性送入 TinyCharEncoder (秒算全部 Z)
                flat_ids = ids_t.view(-1, T)
                flat_att = att_t.view(-1, T)
                
                # B. 计算 Z_target 并重新折叠回 3D 文档结构
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        z_flat = tiny_encoder(flat_ids, attention_mask=flat_att)
                    z_truth = z_flat.view(B, S, -1)
                    # 强行归零 Padding
                    z_truth = torch.where(sen_t[:, :, None] == 0, torch.zeros_like(z_truth), z_truth)
                
                # C. 时空错位法 (Past -> Future)
                f_t_input = z_truth[:, :-1, :]
                z_target_truth = z_truth[:, 1:, :]
                masks_shifted = sen_t[:, 1:]
                
                # D. 物理瞬时解算与反向传播
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = e2e_model(f_t_input, z_target_truth, masks_shifted)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Anomaly] NaN/Inf Loss detected at Step {global_step}! Skipping update.")
                    continue
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(e2e_model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                if global_step % 10 == 0:
                    print(f"Epoch {epoch+1} | Step {global_step} | Flow Match Loss (Kinetic Error): {loss.item():.4f}")
                    
                if global_step % args.save_steps == 0:
                    save_checkpoint(args.out_dir, prefix, global_step, e2e_model, optimizer, dataloader)
                    
    except KeyboardInterrupt:
        save_checkpoint(args.out_dir, prefix, global_step, e2e_model, optimizer, dataloader)

if __name__ == "__main__":
    main()
