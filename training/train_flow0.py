import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import json

from training.core.char_tokenizer import CharTokenizer
from model.config import ModelConfig
from model.adapter import SensoryFuser
from model.god_encoder import GodEncoder
from model.flow_decoder import FlowDecoder
from training.core.dataloader import MultiEmbDataLoader, ChunkedNpzDataLoader
from training.losses.flow_loss import ot_cfm_loss
from training.core.checkpoint import Checkpointer
from training.core.schedule import linear_warmup_schedule
from training.core.args import get_training_parser

def main():
    parser = get_training_parser(description="Phase 0 Flow Matching: GodEncoders + Continuous ODE FlowDecoder")
    args = parser.parse_args()

    # 1. Config & Tokenizer
    tokenizer = CharTokenizer()
    config = ModelConfig()
    config.z_dim = args.z_dim
    config.vocab_size = tokenizer.vocab_size

    # 2. 从独立配置读取 ModelScope 数据流向
    with open(args.config_file, "r", encoding="utf-8") as f:
        ds_config = json.load(f)
        
    base_models = ds_config.get("base_models", ["bge", "qwen", "xiaobu", "youtu", "conan_v1"])
    chunk_patterns = ds_config.get("chunk_name_patterns", {})
    ms_repo_id = ds_config.get("modelscope_repo_id", "catclever/emb_npy")
    parquet_path = ds_config.get("parquet_path", "data/Basic_ZH/chunked_mixed_wiki.parquet")

    # 3. Dataloader (ModelScope Remote Streaming)
    # 彻底弃用本地巨大 emb_files，使用 ChunkedNpzDataLoader
    will_resume = args.auto_resume or args.resume_from
    dataloader = ChunkedNpzDataLoader(
        parquet_path=parquet_path,
        models=base_models,
        chunk_patterns=chunk_patterns,
        tokenizer=tokenizer,
        ms_repo_id=ms_repo_id,
        batch_size=args.batch_size,
        max_seq_len=config.max_seq_len,
        shuffle=True,
        backend='mlx',
        lazy_start=will_resume,  # 恢复时延迟启动，避免白下载一个块
        cache_dir=args.data_dir
    )

    # 3. Models Setup
    d_model = config.decoder_heads * 64
    emb_dims = ds_config.get("emb_dims")
    if not emb_dims:
        # 回退：从 dataloader 首块侦测（需要等下载完成）
        if not hasattr(dataloader, 'emb_dims'):
            raise ValueError("dataset_config.json 中缺少 emb_dims，且 dataloader 尚未加载数据")
        emb_dims = dataloader.emb_dims
    fuser = SensoryFuser(emb_dims, d_model)
    god_encoder = GodEncoder(d_model, config.z_dim)
    decoder = FlowDecoder(config.z_dim, d_model, config.vocab_size, n_layers=config.decoder_layers)
    
    class FlowPhase0Composite(nn.Module):
        def __init__(self, fuser, god_enc, dec):
            super().__init__()
            self.fuser = fuser
            self.god_encoder = god_enc
            self.decoder = dec
            
        def __call__(self, embs, tokens, mask):
            # Stochastic Fusion: 严格按照 配置分配权重
            if self.training:
                import random
                N = len(embs)
                weights = [args.fusion_other_weight] * N
                main_idx = random.randint(0, N - 1)
                weights[main_idx] = args.fusion_main_weight
                f_t = self.fuser(embs, weights=weights)
            else:
                f_t = self.fuser(embs, weights=None)
                
            z_target = self.god_encoder(f_t)
            return z_target
            
    model_composite = FlowPhase0Composite(fuser, god_encoder, decoder)
    mx.eval(model_composite.parameters())
    print("Flow Model composite initialized.")

    # 4. Checkpointer
    checkpointer = Checkpointer(args.out_dir, prefix=args.ckpt_prefix)
    
    checkpointer.register_model("sense_fuser", fuser)
    checkpointer.register_model("god_encoder", god_encoder)
    checkpointer.register_model("flow_decoder", decoder)
    checkpointer.register_dataloader("dataloader", dataloader)
    checkpointer.register_args(args)
    
    start_step = 0
    if args.resume_from:
        start_step = checkpointer.load(args.resume_from)
    elif args.auto_resume:
        start_step = checkpointer.load_latest()

    # 5. Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr)

    # 6. Optimal Transport Flow Match Objective + Auxiliary Losses
    def loss_fn(model, embs, tokens, target_mask):
        z_target = model(embs, tokens, target_mask)
        loss = ot_cfm_loss(model.decoder, tokens, z_target, mask=target_mask,
                           x1_weight=args.x1_weight, 
                           snap_ce_weight=args.snap_ce_weight,
                           t_power=args.t_power)
        return loss

    step_fn = nn.value_and_grad(model_composite, loss_fn)

    # 7. Training Loop 
    print(f"Starting Flow Phase 0 Training. Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    print(f"  Auxiliary losses: x1_weight={args.x1_weight}, snap_ce_weight={args.snap_ce_weight}, t_power={args.t_power}")
    global_step = start_step

    try:
        for epoch in range(dataloader.current_epoch, args.epochs):
            for token_inputs, batch_embs, attention_mask in dataloader:
                
                # Sync LR natively
                optimizer.learning_rate = linear_warmup_schedule(global_step, args.lr, args.warmup_steps)
                
                # Notice: No shifting [:-1] and [1:] needed here! Sequence is treated as a continuous physical block.
                loss, grads = step_fn(model_composite, batch_embs, token_inputs, attention_mask)
                
                optimizer.update(model_composite, grads)
                mx.eval(model_composite.parameters(), optimizer.state)
                
                global_step += 1
                
                if global_step % 10 == 0:
                    print(f"Epoch {epoch+1} | Step {global_step} | L_ot_cfm: {loss.item():.4f}")
                    
                if global_step % args.save_steps == 0:
                    checkpointer.save(global_step)
                    
    except KeyboardInterrupt:
        checkpointer.save(global_step, is_emergency=True)

if __name__ == "__main__":
    main()
