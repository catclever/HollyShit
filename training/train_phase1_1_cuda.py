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
from training.core.adaptive_noise import AdaptiveNoiseScheduler
from training.core.char_tokenizer import CharTokenizer
from training.core.schedule import linear_warmup_schedule
from training.losses.flow_loss_v2_cuda import compute_flow_matching_loss_v2
from training.core.args import get_training_parser

# Dynamic Import of the Phase 0.5 Distilled Model
from distilled_emb.model_cuda import TinyCharEncoderCUDA

class E2EModel(nn.Module):
    """
    联合训练容器：将 Mamba (大脑) 和 FlowMatcher (喷嘴) 缝合，带有各向异性先验
    """
    def __init__(self, mamba, flow, prior_mean, prior_std):
        super().__init__()
        self.mamba = mamba
        self.flow = flow
        # 注册为 buffer，跟随模型设备但不参与梯度更新
        self.register_buffer("prior_mean", prior_mean)
        self.register_buffer("prior_std", prior_std)
        
    def forward(self, main_stream, z_target, mask):
        # 1. 大脑思考：Mamba 吸收历史算子，积分出当前的纯净势能 h_context
        h_context = self.mamba(main_stream, aux_streams=[])
        
        # 2. 嘴巴吹风：FlowMatcher 在 h_context 的势能场下，尝试吹出微观轨迹 (引入各向异性先验)
        loss = compute_flow_matching_loss_v2(
            self.flow, 
            z_target, 
            h_context, 
            prior_mean=self.prior_mean,
            prior_std=self.prior_std,
            mask=mask
        )
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
    if os.path.exists(os.path.join(latest_dir, "dataloader_phase1_1.json")):
        with open(os.path.join(latest_dir, "dataloader_phase1_1.json"), "r") as f:
            dataloader.load_state_dict(json.load(f))
            
    return max_step

def save_checkpoint(out_dir, prefix, step, model, optimizer, dataloader, keep_last_k=5):
    import shutil
    import glob
    import re
    
    dir_name = f"{prefix}_step_{step}" if prefix else f"step_{step}"
    save_path = os.path.join(out_dir, dir_name)
    os.makedirs(save_path, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_path, "e2e_model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
    
    with open(os.path.join(save_path, "dataloader_phase1_1.json"), "w") as f:
        json.dump(dataloader.state_dict(), f)
    print(f"Saved checkpoint to {save_path}")
    
    # Cleanup old checkpoints
    if keep_last_k > 0:
        dirs = glob.glob(os.path.join(out_dir, f"{prefix}_step_*" if prefix else "step_*"))
        step_dirs = []
        for d in dirs:
            m = re.search(r'step_(\d+)', d)
            if m:
                step_dirs.append((int(m.group(1)), d))
        step_dirs.sort(key=lambda x: x[0])
        
        while len(step_dirs) > keep_last_k:
            oldest_step, oldest_dir = step_dirs.pop(0)
            shutil.rmtree(oldest_dir, ignore_errors=True)
            print(f"Removed old checkpoint {oldest_dir}")

def main():
    parser = get_training_parser("Phase 1.1: Mamba & Flow Matching Training with Anisotropic Prior (CUDA Version)")
    parser.add_argument("--max_episode_len", type=int, default=None, help="Sequence chunking limit.")
    parser.add_argument("--data_path", type=str, default="data/Basic_ZH/chunked_mixed_omni.parquet", help="Path to the parquet training data.")
    parser.add_argument("--tinybert_ckpt", type=str, default="checkpoints/distilled/tinybert_pt_v1_step_100000", help="Path to frozen Phase 0.5 distilled TinyBERT.")
    parser.add_argument("--d_model", type=int, default=1024, help="Dimension of the Mamba and FlowMatcher physical backbone.")
    parser.add_argument("--mamba_d_state", type=int, default=16, help="Mamba internal state dimension.")
    parser.add_argument("--mamba_d_conv", type=int, default=4, help="Mamba internal conv dimension.")
    parser.add_argument("--mamba_expand", type=int, default=2, help="Mamba internal expansion factor.")
    parser.add_argument("--flow_hidden_dim", type=int, default=2048, help="Hidden expansion dimension for the Flow Matcher nozzle.")
    
    # Phase 1.1 Specific Argument
    parser.add_argument("--z_stats", type=str, default="checkpoints/z_space_stats.pt", help="Path to precomputed Z space statistics.")
    
    # Noise Scheduler Arguments
    parser.add_argument("--noise_warmup_steps", type=int, default=200000, help="Global steps before noise is injected")
    parser.add_argument("--max_noise_ratio", type=float, default=0.5, help="Maximum mean noise ratio (R_max)")
    parser.add_argument("--noise_variance", type=float, default=0.01, help="Variance of the noise ratio distribution")
    parser.add_argument("--noise_loss_low", type=float, default=0.35, help="Loss lower bound for max noise")
    parser.add_argument("--noise_loss_high", type=float, default=0.6, help="Loss upper bound for zero noise")
    
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
    
    ckpt_path = args.tinybert_ckpt
    if not ckpt_path.endswith(".pt") and not ckpt_path.endswith(".safetensors"):
        if os.path.exists(f"{ckpt_path}/student.pt"):
            ckpt_path = f"{ckpt_path}/student.pt"
        elif os.path.exists(f"{ckpt_path}/model.safetensors"):
            ckpt_path = f"{ckpt_path}/model.safetensors"
        else:
            raise FileNotFoundError(f"Could not find student.pt or model.safetensors in {args.tinybert_ckpt}.")

    if ckpt_path.endswith(".safetensors"):
        weights = load_file(ckpt_path)
    else:
        weights = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        
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
    
    # 2.5 加载预先统计的 Z 空间规范参数 (Phase 1.1 核心增量)
    if not os.path.exists(args.z_stats):
        raise FileNotFoundError(f"Missing {args.z_stats}! Please run tools/extract_z_space_stats.py first.")
    
    print(f"Loading Z-space statistics from {args.z_stats}...")
    stats = torch.load(args.z_stats, map_location='cpu')
    prior_mean = stats['mean'].to(device)
    prior_std = stats['std'].to(device)
    # 给一个小小的底噪保护，防止由于某些维度几乎为 0 导致后续网络梯度爆炸
    prior_std = torch.clamp(prior_std, min=1e-4) 
    print("Successfully loaded Anisotropic Gaussian Prior!")

    # 3. 实例化真正的训练目标：Mamba大脑 + 喷嘴
    mamba_planner = MambaPlanner(
        d_model=d_model, 
        d_state=args.mamba_d_state,
        d_conv=args.mamba_d_conv,
        expand=args.mamba_expand
    )
    flow_matcher = FlowMatcher(d_model=d_model, hidden_dim=args.flow_hidden_dim)
    
    e2e_model = E2EModel(mamba_planner, flow_matcher, prior_mean, prior_std).to(device)
    
    optimizer = optim.AdamW(e2e_model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler() # AMP 加速

    # 4. 恢复状态
    prefix = args.ckpt_prefix if args.ckpt_prefix else "p1_1_flow_cuda"
    start_step = 0
    if args.resume_from or args.auto_resume:
        load_target = args.resume_from if args.resume_from else args.out_dir
        start_step = load_checkpoint(load_target, prefix, e2e_model, optimizer, dataloader, device)

    print(f"Starting Phase 1.1 E2E Physics Training on CUDA. Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    global_step = start_step

    # Initialize Adaptive Noise Scheduler
    noise_scheduler = AdaptiveNoiseScheduler(
        warmup_steps=args.noise_warmup_steps,
        max_noise_ratio=args.max_noise_ratio,
        noise_variance=args.noise_variance,
        loss_range=(args.noise_loss_low, args.noise_loss_high)
    )

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
                
                # C. 时空错位法 (Past -> Future) 与 动态加噪注入
                f_t_input_clean = z_truth[:, :-1, :]
                f_t_input, noise_ratio = noise_scheduler.inject_noise(f_t_input_clean, global_step)
                
                z_target_truth = z_truth[:, 1:, :]
                masks_shifted = sen_t[:, 1:]
                
                # D. 物理瞬时解算与反向传播
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = e2e_model(f_t_input, z_target_truth, masks_shifted)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Anomaly] NaN/Inf Loss detected at Step {global_step}! Skipping update.")
                    continue
                    
                noise_scheduler.step(loss.item())
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(e2e_model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                if global_step % 10 == 0:
                    print(f"Epoch {epoch+1} | Step {global_step} | Flow Loss: {loss.item():.4f} | Noise Ratio: {noise_ratio:.3f} | EMA Loss: {noise_scheduler.ema_loss:.3f}")
                    
                if global_step % args.save_steps == 0:
                    save_checkpoint(args.out_dir, prefix, global_step, e2e_model, optimizer, dataloader, keep_last_k=args.keep_last_k)
                    
    except KeyboardInterrupt:
        save_checkpoint(args.out_dir, prefix, global_step, e2e_model, optimizer, dataloader, keep_last_k=args.keep_last_k)

if __name__ == "__main__":
    main()
