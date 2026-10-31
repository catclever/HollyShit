import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Ensure imports work regardless of execution directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path and os.path.exists(os.path.join(script_dir, "model.py")):
    sys.path = [p for p in sys.path if p != script_dir] + [script_dir]

from training.core.char_tokenizer import CharTokenizer
from training.data.dialogue_dataset import GenericPairDataset
from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from model.continuous_translator_cuda import ContinuousToTextTranslatorCUDA
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer
from torch.utils.data import DataLoader

def decode_with_eos_marker(tokenizer, token_ids):
    if tokenizer.eos_token_id in token_ids:
        eos_idx = token_ids.index(tokenizer.eos_token_id)
        head = tokenizer.decode(token_ids[:eos_idx], skip_special_tokens=True)
        tail = tokenizer.decode(token_ids[eos_idx + 1:], skip_special_tokens=True)
        return f"{head}<eos>{tail}"
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def synthesize_degraded_manifold_curve(x_target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Synthesizes realistic ODE + Legendre manifold degradations (neighbor bleed + isotropic Gaussian jitter)
    so the 1D CNN Translator (`Conv1d(kernel_size=3)`) explicitly learns:
      1. High-pass Laplacian de-blurring (un-mixing adjacent characters to prevent AABB stuttering);
      2. Wide Voronoi attraction basins around each character coordinate for ODE landing tolerance.
    """
    B, L, D = x_target.shape
    mask_f = valid_mask.unsqueeze(-1).to(x_target.dtype)

    # 1. Local neighbor bleed simulation: alpha in [0.05, 0.30] per sample
    if L > 2:
        x_left = torch.zeros_like(x_target)
        x_right = torch.zeros_like(x_target)
        x_left[:, 1:, :] = x_target[:, :-1, :] * mask_f[:, :-1, :]
        x_right[:, :-1, :] = x_target[:, 1:, :] * mask_f[:, 1:, :]
        x_neigh = 0.5 * (x_left + x_right)
        alpha = torch.empty((B, 1, 1), device=x_target.device, dtype=x_target.dtype).uniform_(0.05, 0.30)
        x_blur = (1.0 - alpha * mask_f) * x_target + (alpha * mask_f) * x_neigh
    else:
        x_blur = x_target

    # 2. Isotropic manifold ODE jitter: sigma in [0.05, 0.30] * valid_std
    valid_tokens = x_target[valid_mask] if valid_mask.any() else x_target
    valid_std = valid_tokens.float().std().clamp(min=1e-4).to(x_target.dtype)
    sigma = torch.empty((B, 1, 1), device=x_target.device, dtype=x_target.dtype).uniform_(0.05, 0.30)
    noise = torch.randn_like(x_target) * (sigma * valid_std)

    return x_blur + noise


def train_translator():
    parser = get_training_parser("Phase 4 Sharpening & Denoising CNN Translator Training (Pure Text Manifold)")
    parser.add_argument("--data_path", type=str, default="data/Basic_ZH/moss_dialogues.jsonl", help="Path to pure dialogue/text dataset (no knowledge required)")
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/basencoder_step_115000/student.pt", help="Path to frozen Phase 1 encoder checkpoint")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to load")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional max training steps (e.g. 3000 for fast convergence)")
    parser.add_argument("--min_target_len", type=int, default=10, help="Minimum clause character length after punctuation split")
    parser.add_argument("--max_clause_chars", type=int, default=28, help="Maximum clause character length after punctuation split")
    parser.add_argument("--split_by_punctuation", action="store_true", default=True, help="Split targets into 10-28 char punctuation clauses")
    parser.add_argument("--no_split_by_punctuation", dest="split_by_punctuation", action="store_false", help="Disable punctuation splitting")
    parser.add_argument("--deblur_weight", type=float, default=0.5, help="Weight of de-blurring/denoising branch loss in [0, 1]")
    parser.set_defaults(
        ckpt_prefix="translator_conv1d",
        seq_len=32,
        batch_size=192,
        epochs=1,
        save_steps=1000,
    )
    args = parser.parse_args()
    if not args.ckpt_prefix:
        args.ckpt_prefix = "translator_conv1d"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = CharTokenizer()
    # 纯文本声码器训练（默认 MOSS 对话语料，无需任何知识库输入，仅对齐 seq_len=32 标点短分句长度）
    dataset = GenericPairDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        prompt_seq_len=16,
        target_seq_len=args.seq_len,
        max_samples=args.max_samples,
        min_target_len=args.min_target_len,
        split_by_punctuation=args.split_by_punctuation,
        max_clause_chars=args.max_clause_chars,
        min_clause_chars=args.min_target_len,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 1. 挂载并彻底冻结 Phase 1 Encoder
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    ckpt_path = args.encoder_ckpt
    if os.path.exists(ckpt_path):
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"Loaded frozen encoder from {ckpt_path}")
    else:
        print(f"WARNING: Encoder checkpoint {ckpt_path} not found. Using random weights (for testing only).")
    
    encoder.eval()
    for p in encoder.parameters(): 
        p.requires_grad = False
        
    # 2. 初始化 Translator (LM Head)
    translator = ContinuousToTextTranslatorCUDA(d_model=1024, vocab_size=tokenizer.vocab_size).to(device)
    
    optimizer = optim.AdamW(translator.parameters(), lr=1e-3)
    # 对 <PAD> 赋予 0.25 权重，确保 seq_len=32 下 <EOS> 之后的空白流形稳定收敛回 <PAD> (ID=0)，彻底消除尾部乱码
    class_weights = torch.ones(tokenizer.vocab_size, device=device)
    class_weights[tokenizer.pad_token_id] = 0.25
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=3)
    checkpointer.register_model("translator", translator)
    checkpointer.register_optimizer("opt", optimizer)
    
    global_step = checkpointer.load_latest() if args.auto_resume else 0
    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    
    print(
        f"Starting Sharpening & Denoising Translator Training "
        f"(seq_len={args.seq_len}, deblur_weight={args.deblur_weight:.2f})..."
    )
    
    for epoch in range(start_epoch, args.epochs):
        resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        translator.train()
        
        for batch_idx, batch_data in enumerate(resumed_loader):
            w_tgt = batch_data[-1].to(device)  # (B, L) - ground truth discrete tokens
            valid_mask = (w_tgt != tokenizer.pad_token_id)
            
            # Step 1: Forward through frozen encoder to get Ground Truth Continuous Curve
            with torch.no_grad():
                h_tgt = encoder.get_normalized_embeddings(w_tgt)
                for layer in encoder.layers: 
                    h_tgt = layer(h_tgt, None)
                x_target = encoder.final_ln(h_tgt)  # (B, L, 1024)
                x_degraded = synthesize_degraded_manifold_curve(x_target, valid_mask)
            
            # Step 2: Dual Clean + De-blurring/Denoising prediction
            optimizer.zero_grad()
            logits_clean = translator(x_target)
            loss_clean = criterion(logits_clean.view(-1, tokenizer.vocab_size), w_tgt.view(-1))

            w_deb = float(args.deblur_weight)
            if w_deb > 0.0:
                logits_deg = translator(x_degraded)
                loss_deg = criterion(logits_deg.view(-1, tokenizer.vocab_size), w_tgt.view(-1))
                loss = (1.0 - w_deb) * loss_clean + w_deb * loss_deg
            else:
                loss_deg = loss_clean
                loss = loss_clean
            
            loss.backward()
            optimizer.step()
            
            global_step += 1
            if global_step % 50 == 0:
                print(
                    f"Epoch {epoch+1} | Step {global_step} | "
                    f"Loss: {loss.item():.4f} (Clean: {loss_clean.item():.4f}, Deblur: {loss_deg.item():.4f})"
                )
                
            if global_step == 50 or global_step % 200 == 0:
                # Quick qualitative test on both Clean and Degraded curves
                translator.eval()
                with torch.no_grad():
                    pred_clean = translator.decode_greedy(x_target[:1])[0].tolist()
                    pred_deg = translator.decode_greedy(x_degraded[:1])[0].tolist()
                    print("\n--- Quick Translation Test (seq_len=32) ---")
                    print(f"Target       : {decode_with_eos_marker(tokenizer, w_tgt[0].tolist())}")
                    print(f"Decode Clean : {decode_with_eos_marker(tokenizer, pred_clean)}")
                    print(f"Decode Deblur: {decode_with_eos_marker(tokenizer, pred_deg)}")
                    print("-------------------------------------------\n")
                translator.train()

            if args.save_steps and global_step % args.save_steps == 0:
                checkpointer.save(global_step)

            if args.max_steps and global_step >= args.max_steps:
                print(f"✅ Reached max_steps={args.max_steps}. Saving final translator checkpoint...")
                checkpointer.save(global_step)
                return
                
        checkpointer.save(global_step)

if __name__ == "__main__":
    train_translator()
