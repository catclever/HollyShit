# =====================================================================================================
# [Task #033: Phase 5 Context Knowledge-Selector Trainer (independent of the right brain)]
# 1. What: trains ONLY `KnowledgeSelector` (incl. the #033 sink / "copy nothing" head) on the #032
#    clause-stream data. For every supervised row t the candidate pool is every earlier valid row j < t of
#    the same main stream; labels are the IDF-weighted rare-span overlap pseudo labels `ctx_rel (T, T)` and
#    `sink_label` (= no earlier row reaches --rel_threshold). Encoder / left brain / HiPPO (+ translator for
#    eval, + optional right brain for probes) are frozen.
# 2. Why: staged plan §3 (design/tentative/2026-11-02_phase5_staged_training_plan.md): the selector is
#    trained independently of the right brain; #031 showed irrelevant attractors hurt the left brain
#    (B < C), so the selector must learn when to pass NO attractors (sink) -> `k_attractors=None`.
#    Probe = ground-truth target clause vector + probe_noise_std * row_std * N(0, 1); optionally (stage-2
#    style fine-tuning) replaced with prob --p_rb_probe by right-brain sampled thoughts (teacher-forced
#    context, no_grad) from --right_brain_ckpt so the selector sees inference-time probes.
# 3. Eval: N fixed benchmark rows; the target clause is decoded by the frozen left brain from its GT
#    vector under 3 attractor settings sharing the same initial noise: selector (sink-aware), none,
#    oracle (top-M rows with ctx_rel >= --rel_threshold, none if no such row). Reports char accuracy,
#    edit similarity, rare-span recall (needs translator), latent cosine to the GT char embeddings (always),
#    sink fire rate / agreement with sink_label, and how often selection beats none.
# 4. Caveats:
#    - Shared helpers come from the Stage-1 trainer module `training.train_phase5_right_brain_stage1`.
#    - Checkpoints: `knowledge_selector.pt` (+ optimizer / scheduler), prefix `selector_ctx`. Load the result
#      in the Stage-1 eval with `--eval_attractors context --selector_ckpt <dir>`.
#    - precision@1 / recall@M use positives = rel >= --rel_threshold (fallback: the row-max candidate);
#      keep --rel_threshold equal to the dataset's threshold so metrics match the sink definition.
#    - Without a translator only the latent cosine metric is meaningful.
# =====================================================================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time
import difflib
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.core.char_tokenizer import CharTokenizer
from training.core.args import get_training_parser
from training.core.checkpoint import Checkpointer

from model.bicameral_right_brain import KnowledgeSelector, crystallize_with_optional_attractors
from training.train_phase5_right_brain_stage1 import (
    SOURCE_NAMES,
    Profiler,
    _fmt_opt,
    _scalar,
    add_data_args,
    add_frozen_ckpt_args,
    build_dataloader,
    decode_clause_clean,
    decode_x,
    encode_clause_bank_to_2d,
    encode_tokens,
    get_warmup_cosine_scheduler,
    load_frozen_modules,
    load_right_brain,
    resume_training,
    select_benchmark_items,
    setup_amp,
)


def _char_acc(gt: str, pred: str) -> float:
    if not gt:
        return 100.0 if not pred else 0.0
    return 100.0 * sum(1 for a, b in zip(gt, pred) if a == b) / float(max(len(gt), len(pred)))


def _edit_sim(gt: str, pred: str) -> float:
    if not gt and not pred:
        return 100.0
    return 100.0 * difflib.SequenceMatcher(None, gt, pred).ratio()


def build_eval_rows(
    bench: List[Tuple[str, Dict[str, Any]]], max_rows: int,
) -> List[Tuple[str, Dict[str, Any], int]]:
    """Round-robin over benchmark items, taking supervised rows (ctx_rel_valid & main_valid, t >= 1)."""
    per_item: List[List[Tuple[str, Dict[str, Any], int]]] = []
    for tag, item in bench:
        mv = item["main_valid"].bool()
        rv = item["ctx_rel_valid"].bool() & mv
        rows = [int(t) for t in torch.nonzero(rv, as_tuple=False).reshape(-1).tolist() if int(t) >= 1]
        per_item.append([(tag, item, t) for t in rows])
    out: List[Tuple[str, Dict[str, Any], int]] = []
    k = 0
    while len(out) < max_rows and any(k < len(r) for r in per_item):
        for r in per_item:
            if k < len(r) and len(out) < max_rows:
                out.append(r[k])
        k += 1
    return out


@torch.no_grad()
def evaluate_selector(
    step: int,
    selector: KnowledgeSelector,
    unet,
    encoder,
    hippo,
    translator,
    tokenizer: CharTokenizer,
    dataset,
    eval_rows: List[Tuple[str, Dict[str, Any], int]],
    args,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    selector.eval()
    pad = tokenizer.pad_token_id
    settings = ("selector", "none", "oracle")
    agg: Dict[str, Dict[str, List[float]]] = {s: {"char_acc": [], "edit_sim": [], "rare_recall": [], "latent_cos": []} for s in settings}
    n_sink_fire, n_sink_agree, n_rows = 0, 0, 0
    n_oracle_none = 0
    beats, deltas = 0, []
    examples: List[str] = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    has_rare = hasattr(dataset, "rare_spans")

    for start in range(0, len(eval_rows), max(1, args.eval_batch)):
        chunk = eval_rows[start:start + max(1, args.eval_batch)]
        Bc = len(chunk)
        w_all = torch.stack([it["w_main"] for _, it, _ in chunk]).to(device)            # (Bc, T, L)
        mv_all = torch.stack([it["main_valid"] for _, it, _ in chunk]).to(device).bool()
        rel_all = torch.stack([it["ctx_rel"] for _, it, _ in chunk]).to(device).float()  # (Bc, T, T)
        sink_all = torch.stack([it["sink_label"] for _, it, _ in chunk]).to(device).float()
        rows = torch.tensor([t for _, _, t in chunk], device=device, dtype=torch.long)
        T, L = w_all.shape[1], w_all.shape[2]
        z_all = encode_clause_bank_to_2d(encoder, unet, hippo, w_all, mv_all, pad).float()  # (Bc, T, D)
        ar = torch.arange(Bc, device=device)
        z_t = z_all[ar, rows]                      # (Bc, D)
        w_t = w_all[ar, rows]                      # (Bc, L)
        cand_valid = (torch.arange(T, device=device).unsqueeze(0) < rows.unsqueeze(1)) & mv_all  # (Bc, T)
        rel_t = rel_all[ar, rows]                  # (Bc, T)
        sink_t = sink_all[ar, rows] > 0.5

        # selector (sink-aware)
        w_sel, k_sel, info = selector.select_context_attractors(
            z_t, z_all, w_all, cand_valid, top_m=args.top_m, min_score=args.min_selector_score,
            sink_threshold=args.sink_threshold, pad_token_id=pad,
        )
        use_sel = info["use_attr"]
        # oracle: top-M rows by ctx_rel >= rel_threshold (chronological), none if no such row
        M = max(1, min(args.top_m, T))
        rel_m = torch.where(cand_valid, rel_t, torch.full_like(rel_t, -1.0))
        top_r, top_i = rel_m.topk(M, dim=-1)
        keep = top_r >= float(args.rel_threshold)
        key = torch.where(keep, top_i, top_i + T)
        order = key.argsort(dim=-1)
        top_i, keep = torch.gather(top_i, 1, order), torch.gather(keep, 1, order)
        use_ora = keep.any(dim=-1)
        w_ora = torch.gather(w_all, 1, top_i.unsqueeze(-1).expand(Bc, M, L))
        w_ora = torch.where(keep.unsqueeze(-1), w_ora, torch.full_like(w_ora, pad)).reshape(Bc, M * L)
        k_ora = w_ora != pad

        x0 = torch.randn((Bc, args.seq_len, 1024), generator=gen).to(device)
        x_tgt = encode_tokens(encoder, w_t)        # (Bc, L, D) encoder space
        tok_valid = w_t != pad
        gt_texts = [decode_clause_clean(tokenizer, w_t[i].tolist()) for i in range(Bc)]
        outs: Dict[str, List[str]] = {}
        for s in settings:
            if s == "selector":
                w_a, k_a, u_a = w_sel, k_sel, use_sel
            elif s == "oracle":
                w_a, k_a, u_a = (w_ora, k_ora, use_ora) if bool(use_ora.any()) else (None, None, None)
            else:
                w_a, k_a, u_a = None, None, None
            x_a = unet.normalize(encode_tokens(encoder, w_a)) if w_a is not None else None
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                x_gen = crystallize_with_optional_attractors(
                    unet, z_t, x_a, k_a, u_a, steps=args.left_ode_steps, seq_len=args.seq_len, x0=x0,
                ).float()
            texts, _, _ = decode_x(x_gen, translator, tokenizer)
            outs[s] = texts
            Lm = min(x_gen.shape[1], x_tgt.shape[1])
            cos = F.cosine_similarity(x_gen[:, :Lm], x_tgt[:, :Lm].float(), dim=-1)  # (Bc, Lm)
            vm = tok_valid[:, :Lm].float()
            lat = (cos * vm).sum(-1) / vm.sum(-1).clamp(min=1.0)
            for i in range(Bc):
                agg[s]["latent_cos"].append(float(lat[i].item()))
                if translator is not None:
                    agg[s]["char_acc"].append(_char_acc(gt_texts[i], texts[i]))
                    agg[s]["edit_sim"].append(_edit_sim(gt_texts[i], texts[i]))
                    if has_rare:
                        spans = dataset.rare_spans(gt_texts[i])
                        if spans:
                            agg[s]["rare_recall"].append(100.0 * sum(1 for sp in spans if sp in texts[i]) / len(spans))
        for i in range(Bc):
            n_rows += 1
            fired = not bool(use_sel[i])
            n_sink_fire += int(fired)
            n_sink_agree += int(fired == bool(sink_t[i]))
            n_oracle_none += int(not bool(use_ora[i]))
            key_metric = "edit_sim" if translator is not None else "latent_cos"
            d = agg["selector"][key_metric][-Bc + i] - agg["none"][key_metric][-Bc + i]
            deltas.append(d)
            beats += int(d > 0)
            if len(examples) < args.eval_show:
                tag = chunk[i][0]
                sp = float(info["sink_prob"][i])
                examples.append(
                    f"   [{tag} r{int(rows[i])}] sink_lbl={int(sink_t[i])} sink_p={sp:.2f} use_sel={int(bool(use_sel[i]))} "
                    f"use_ora={int(bool(use_ora[i]))}\n      GT: {gt_texts[i]}\n      sel: {outs['selector'][i]}\n"
                    f"      none: {outs['none'][i]}\n      ora: {outs['oracle'][i]}"
                )

    def _m(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    print("\n" + "=" * 96)
    print(f"🔎 [Step {step}] Selector eval on {n_rows} rows (LeftODE={args.left_ode_steps}, top_m={args.top_m}, "
          f"min_score={args.min_selector_score}, sink_thr={args.sink_threshold}, rel_thr={args.rel_threshold})")
    for s in settings:
        a = agg[s]
        print(f"   {s:<8} | char_acc {_fmt_opt(_m(a['char_acc']), '{:.2f}')} | edit_sim {_fmt_opt(_m(a['edit_sim']), '{:.2f}')} | "
              f"rare_recall {_fmt_opt(_m(a['rare_recall']), '{:.2f}')} | latent_cos {_fmt_opt(_m(a['latent_cos']), '{:.4f}')}")
    if n_rows:
        print(f"   sink fired {n_sink_fire}/{n_rows} ({100.0 * n_sink_fire / n_rows:.1f}%) | agrees with sink_label "
              f"{100.0 * n_sink_agree / n_rows:.1f}% | oracle 'none' rows {n_oracle_none}/{n_rows} | "
              f"selector beats none on {beats}/{n_rows} rows (mean Δ {_fmt_opt(_m(deltas), '{:+.3f}')} "
              f"{'edit_sim' if translator is not None else 'latent_cos'})")
    for ex in examples:
        print(ex)
    print("=" * 96 + "\n")
    selector.train()
    return {"edit_sim_selector": _m(agg["selector"]["edit_sim"]) or 0.0, "edit_sim_none": _m(agg["none"]["edit_sim"]) or 0.0}


def build_parser():
    parser = get_training_parser("Phase 5 context KnowledgeSelector training with sink head (#033, CUDA)")
    add_data_args(parser)
    add_frozen_ckpt_args(parser)
    parser.add_argument("--max_steps", type=int, default=20000, help="Training horizon (steps)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Cosine LR floor ratio")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "no"])
    parser.add_argument("--auto_resume", dest="auto_resume", action="store_true", help="Alias of --continue")
    parser.add_argument("--d_proj", type=int, default=256, help="Selector low-rank projection width")
    parser.add_argument("--probe_noise_std", type=float, default=0.05, help="Relative Gaussian probe noise (x row std)")
    parser.add_argument("--right_brain_ckpt", type=str, default=None, help="Frozen Stage-1 right brain for thought probes")
    parser.add_argument("--p_rb_probe", type=float, default=0.5, help="P(replace a GT probe by a right-brain thought)")
    parser.add_argument("--rb_temperature", type=float, default=1.0, help="Right-brain probe sampling temperature")
    parser.add_argument("--rb_ode_steps", type=int, default=3, help="Right-brain probe ODE steps")
    parser.add_argument("--sink_weight", type=float, default=1.0, help="Weight of the sink BCE")
    parser.add_argument("--max_pos_weight", type=float, default=5.0, help="Candidate BCE pos_weight clamp")
    parser.add_argument("--top_m", type=int, default=3, help="Max attractor clauses / recall@M")
    parser.add_argument("--min_selector_score", type=float, default=0.25, help="Min sigmoid score of a selected clause")
    parser.add_argument("--sink_threshold", type=float, default=0.5, help="sink_prob above this -> no attractors")
    parser.add_argument("--eval_interval", type=int, default=500, help="Eval every N steps (0 = off)")
    parser.add_argument("--eval_at_start", action="store_true", help="Run one eval before training")
    parser.add_argument("--log_interval", type=int, default=20, help="Log every N steps")
    parser.add_argument("--eval_rows", type=int, default=24, help="Benchmark rows decoded per eval")
    parser.add_argument("--eval_batch", type=int, default=8, help="Rows per left-brain decode batch")
    parser.add_argument("--eval_show", type=int, default=4, help="Example rows printed per eval")
    parser.add_argument("--bench_qa", type=int, default=6, help="QA samples in the benchmark pool")
    parser.add_argument("--bench_cont", type=int, default=6, help="Continuation samples in the benchmark pool")
    parser.add_argument("--left_ode_steps", "--euler_steps", dest="left_ode_steps", type=int, default=10)
    parser.add_argument("--profile", action="store_true", help="CUDA-synchronised per-module timing")
    parser.set_defaults(seq_len=32, batch_size=32, lr=3e-4, warmup_steps=300, save_steps=1000, ckpt_prefix="selector_ctx")
    return parser


def train():
    args = build_parser().parse_args()
    args.seq_len = min(args.seq_len, 32) if args.seq_len > 64 else args.seq_len
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    amp_enabled, amp_dtype, scaler = setup_amp(args.mixed_precision, device)
    tokenizer = CharTokenizer()
    pad_id = tokenizer.pad_token_id

    dataset, dataloader = build_dataloader(args, tokenizer, shuffle=True)
    bench = select_benchmark_items(dataset, n_qa=args.bench_qa, n_cont=args.bench_cont, seed=42)
    eval_rows = build_eval_rows(bench, args.eval_rows)
    print(f"📌 Selector benchmark: {len(bench)} samples -> {len(eval_rows)} rows")

    encoder, hippo, unet, translator = load_frozen_modules(args, tokenizer, device, need_translator=True)
    right_brain = load_right_brain(args.right_brain_ckpt, device) if args.right_brain_ckpt else None

    selector = KnowledgeSelector(d_clause=1024, d_proj=args.d_proj).to(device)
    optimizer = optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=0.01)
    horizon = args.max_steps if args.max_steps > 0 else max(1, args.epochs * len(dataloader))
    scheduler = get_warmup_cosine_scheduler(optimizer, args.warmup_steps, horizon, args.min_lr_ratio)
    checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix, keep_last_k=args.keep_last_k)
    checkpointer.register_model("knowledge_selector", selector)
    checkpointer.register_optimizer("selector_opt", optimizer)
    checkpointer.register_optimizer("lr_sched", scheduler)
    checkpointer.register_args(args)
    global_step = resume_training(args, checkpointer, {"knowledge_selector": selector}, scheduler, device)

    print(f"🚀 Context KnowledgeSelector (#033) | params {sum(p.numel() for p in selector.parameters()) / 1e6:.2f}M | "
          f"probe_noise={args.probe_noise_std} | rb_probe={'on p=' + str(args.p_rb_probe) if right_brain is not None else 'off'} | "
          f"rel_thr={args.rel_threshold} | sink_w={args.sink_weight} | ckpt_prefix={args.ckpt_prefix}")

    prof = Profiler(args.profile, device)

    def run_eval(step: int) -> float:
        t0 = time.time()
        if eval_rows:
            evaluate_selector(step, selector, unet, encoder, hippo, translator, tokenizer, dataset, eval_rows,
                              args, device, amp_enabled, amp_dtype)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        dt = time.time() - t0
        if args.profile:
            print(f"    ⏱ eval wall time {dt:.2f}s")
        return dt

    if args.eval_at_start:
        run_eval(global_step)

    start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
    last_log_time, last_log_step, excluded = time.time(), global_step, 0.0
    done = False
    selector.train()
    for epoch in range(start_epoch, args.epochs):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        loader = checkpointer.wrap_dataloader(dataloader, epoch, global_step)
        for batch in loader:
            prof.start()
            w_main = batch["w_main"].to(device)
            main_valid = batch["main_valid"].to(device).bool()
            ctx_rel = batch["ctx_rel"].to(device).float()
            ctx_rel_valid = batch["ctx_rel_valid"].to(device).bool()
            sink_label = batch["sink_label"].to(device).float()
            source_id = batch["source_id"].to(device).long()  # logging only
            prof.mark("data")
            with torch.no_grad():
                z_main = encode_clause_bank_to_2d(encoder, unet, hippo, w_main, main_valid, pad_id).float()
                prof.mark("encode")
                row_std = z_main.std(dim=-1, keepdim=True)
                z_probe = z_main + float(args.probe_noise_std) * row_std * torch.randn_like(z_main)
                n_rb = 0
                if right_brain is not None and args.p_rb_probe > 0 and z_main.shape[1] >= 2:
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        thoughts = right_brain.sample_next_clauses(
                            z_main[:, :-1], temperature=args.rb_temperature, ode_steps=args.rb_ode_steps,
                        ).float()  # thoughts[:, t-1] = sampled row t given rows < t
                    rep = (torch.rand(main_valid[:, 1:].shape, device=device) < args.p_rb_probe) & main_valid[:, 1:]
                    z_probe[:, 1:] = torch.where(rep.unsqueeze(-1), thoughts, z_probe[:, 1:])
                    n_rb = int(rep.sum().item())
                prof.mark("probe")
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss, m = selector.compute_context_selection_loss(
                    z_main, z_probe, ctx_rel, ctx_rel_valid, sink_label, main_valid,
                    top_m=args.top_m, pos_threshold=args.rel_threshold,
                    max_pos_weight=args.max_pos_weight, sink_weight=args.sink_weight,
                )
            prof.mark("selector_fwd")
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            prof.mark("backward")
            prof.end_step()

            global_step += 1
            checkpointer.set_step(global_step)
            is_eval = args.eval_interval > 0 and global_step % args.eval_interval == 0
            if global_step == 1 or global_step % max(1, args.log_interval) == 0 or is_eval:
                now = time.time()
                sps = (now - last_log_time - excluded) / max(1, global_step - last_log_step)
                last_log_time, last_log_step, excluded = now, global_step, 0.0
                n_qa = int((source_id == 0).sum().item())
                print(
                    f"Epoch {epoch + 1} | Step {global_step} | LR {optimizer.param_groups[0]['lr']:.2e} | {sps:.2f}s/step | "
                    f"Loss {m['sel_loss']:.4f} (cand {m['cand_loss']:.4f}, sink {m['sink_loss']:.4f}, pw {m['pos_weight']:.2f}) | "
                    f"P@1 {_fmt_opt(m['precision_at_1'])} R@{args.top_m} {_fmt_opt(m['recall_at_m'])} "
                    f"(rank rows {m['n_rank_rows']}) | sink acc {_fmt_opt(m['sink_acc'])} rate pred/true "
                    f"{_fmt_opt(m['sink_rate_pred'])}/{_fmt_opt(m['sink_rate_true'])} | rows {m['n_rows']} | "
                    f"qa/cont {n_qa}/{int(source_id.numel()) - n_qa} | rb probes {n_rb} | "
                    f"scale {float(selector.logit_scale.clamp(1, 30).item()):.2f} bias {float(selector.bias.item()):.2f}"
                )
                rep_s = prof.report_and_reset()
                if rep_s:
                    print(rep_s)
            if is_eval:
                excluded += run_eval(global_step)
            if args.save_steps and global_step % args.save_steps == 0:
                t0 = time.time()
                checkpointer.save(global_step)
                excluded += time.time() - t0
            if args.max_steps > 0 and global_step >= args.max_steps:
                done = True
                break
        if done:
            break

    if not (args.save_steps and global_step > 0 and global_step % args.save_steps == 0):
        checkpointer.save(global_step)
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
