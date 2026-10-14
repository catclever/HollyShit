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
from training.data.moss_dataset import MOSSDataset
from model.phase1_charencoder_cuda import TinyCharEncoderCUDA
from model.continuous_translator_cuda import ContinuousToTextTranslatorCUDA
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer
from torch.utils.data import DataLoader

def train_translator():
    # 复用现有的 training parser，但修改默认后缀
    parser = get_training_parser("Phase 4 Translator (LM Head) Training")
    parser.set_defaults(ckpt_prefix="translator_conv1d")
    args = parser.parse_args()
    if not args.ckpt_prefix:
        args.ckpt_prefix = "translator_conv1d"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = CharTokenizer()
    # 使用专用的 MOSS 数据加载器
    dataset = MOSSDataset(
        jsonl_path="data/Basic_ZH/moss_dialogues.jsonl",
        tokenizer=tokenizer,
        seq_len=args.seq_len
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 1. 挂载并彻底冻结 Phase 1 Encoder
    encoder = TinyCharEncoderCUDA(vocab_size=tokenizer.vocab_size).to(device)
    ckpt_path = "checkpoints/basencoder_step_115000/student.pt"
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
    
    # 这是一个非常简单的纯凸优化问题，Learning Rate 可以开大一点
    optimizer = optim.AdamW(translator.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=3)
    checkpointer.register_model("translator", translator)
    checkpointer.register_optimizer("opt", optimizer)
    
    global_step = checkpointer.load_latest() if args.auto_resume else 0
    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    
    print("Starting Translator Training... (This should converge incredibly fast)")
    
    for epoch in range(start_epoch, args.epochs):
        resumed_loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        translator.train()
        
        for batch_idx, (w_t, w_tgt) in enumerate(resumed_loader):
            w_tgt = w_tgt.to(device) # (B, L) - ground truth discrete tokens
            
            # Step 1: Forward through frozen encoder to get Ground Truth Continuous Curve
            with torch.no_grad():
                h_tgt = encoder.get_normalized_embeddings(w_tgt)
                for layer in encoder.layers: 
                    h_tgt = layer(h_tgt, None)
                x_target = encoder.final_ln(h_tgt) # (B, L, 1024)
            
            # Step 2: Predict logits from the continuous curve
            optimizer.zero_grad()
            logits = translator(x_target) # (B, L, Vocab)
            
            # Compute Loss
            # Flatten for CrossEntropy: (B*L, Vocab) vs (B*L)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), w_tgt.view(-1))
            
            loss.backward()
            optimizer.step()
            
            global_step += 1
            if global_step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {global_step} | Translator Loss: {loss.item():.4f}")
                
            if global_step == 50 or global_step % 200 == 0:
                # Quick qualitative test
                translator.eval()
                with torch.no_grad():
                    pred_ids = translator.decode_greedy(x_target[:1])
                    print("\n--- Quick Translation Test ---")
                    print(f"Target: {tokenizer.decode(w_tgt[0].tolist())}")
                    print(f"Decode: {tokenizer.decode(pred_ids[0].tolist())}")
                    print("------------------------------\n")
                translator.train()
                
        checkpointer.save(global_step)

if __name__ == "__main__":
    train_translator()
