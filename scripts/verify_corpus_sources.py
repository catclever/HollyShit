#!/usr/bin/env python3
"""Verify candidate Phase-5 corpus sources on the Hugging Face Hub.

Probes each candidate dataset (streaming first, then parquet-conversion
fallback, then file listing), records loadability, field structure, samples,
size and license metadata, and runs Hub searches for additional candidates.
Only small samples / metadata are downloaded; nothing is built.

Usage:
    uv run python scripts/verify_corpus_sources.py --out-dir <dir> [--timeout 90]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

N_SAMPLE = 3
N_MRC_STATS = 50
TRUNC = 120

# (category, dataset_id, config or None, alternates[(id, config)])
CANDIDATES: list[dict[str, Any]] = [
    # ---- Long text
    {"cat": "long_text", "id": "0xDing/wikipedia-cn-20230720-filtered", "config": None,
     "alts": [("pleisto/wikipedia-cn-20230720-filtered", None)]},
    {"cat": "long_text", "id": "wikimedia/wikipedia", "config": "20231101.zh", "alts": []},
    {"cat": "long_text", "id": "pleisto/wikipedia-cn-20230720-filtered", "config": None, "alts": []},
    {"cat": "long_text", "id": "wdndev/webnovel-chinese", "config": None, "alts": []},
    {"cat": "long_text", "id": "wangrui6/Zhihu-KOL", "config": None, "alts": []},
    {"cat": "long_text", "id": "BAAI/CCI3-HQ", "config": None, "alts": []},
    # ---- MRC / QA
    {"cat": "mrc", "id": "hfl/cmrc2018", "config": None,
     "alts": [("cmrc2018", None), ("clue/clue", "cmrc2018")]},
    {"cat": "mrc", "id": "cmrc2018", "config": None, "alts": []},
    {"cat": "mrc", "id": "clue", "config": "cmrc2018", "alts": [("clue/clue", "cmrc2018")]},
    {"cat": "mrc", "id": "clue", "config": "drcd", "alts": [("clue/clue", "drcd")]},
    {"cat": "mrc", "id": "clue", "config": "c3", "alts": [("clue/clue", "c3")]},
    {"cat": "mrc", "id": "PaddlePaddle/dureader_robust", "config": None, "alts": []},
    {"cat": "mrc", "id": "luozhouyang/dureader", "config": None, "alts": []},
    # ---- Knowledge dialogue
    {"cat": "kdialog", "id": "thu-coai/kdconv", "config": None, "alts": []},
    {"cat": "kdialog", "id": "PaddlePaddle/duconv", "config": None, "alts": []},
    # ---- Instruction
    {"cat": "instruct", "id": "BelleGroup/train_1M_CN", "config": None, "alts": []},
    {"cat": "instruct", "id": "silk-road/alpaca-data-gpt4-chinese", "config": None, "alts": []},
]

SEARCHES = {
    "mrc": ["dureader", "cmrc", "drcd"],
    "kdialog": ["duconv", "kdconv", "knowledge grounded dialogue chinese",
                "chinese knowledge dialogue", "naturalconv"],
    "instruct": ["COIG", "firefly", "chinese instruction", "alpaca chinese",
                 "chinese system prompt", "ifeval chinese"],
}
# Top search hits per category that get a full probe (in addition to fixed list)
SEARCH_PROBE_TOP = {"mrc": 4, "kdialog": 5, "instruct": 6}


# --------------------------------------------------------------------------- utils
def trunc(v: Any, n: int = TRUNC) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 3] + "..."


def type_tree(v: Any, depth: int = 0) -> Any:
    if depth > 4:
        return type(v).__name__
    if isinstance(v, dict):
        return {k: type_tree(x, depth + 1) for k, x in v.items()}
    if isinstance(v, list):
        if not v:
            return "list[empty]"
        return [type_tree(v[0], depth + 1), f"len={len(v)}"]
    return type(v).__name__


def find_keys(v: Any, names: set[str], prefix: str = "") -> list[str]:
    out = []
    if isinstance(v, dict):
        for k, x in v.items():
            p = f"{prefix}.{k}" if prefix else k
            if k.lower() in names:
                out.append(p)
            out += find_keys(x, names, p)
    elif isinstance(v, list) and v:
        out += find_keys(v[0], names, prefix + "[0]")
    return out


def get_path(rec: Any, path: str) -> Any:
    cur = rec
    for part in path.split("."):
        n_idx = 0
        while part.endswith("[0]"):
            part, n_idx = part[:-3], n_idx + 1
        if part:
            cur = cur.get(part) if isinstance(cur, dict) else None
        for _ in range(n_idx):
            cur = cur[0] if isinstance(cur, list) and cur else None
        if cur is None:
            return None
    return cur


# --------------------------------------------------------------------------- metadata
def hub_meta(ds_id: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    meta: dict[str, Any] = {}
    try:
        info = HfApi().dataset_info(ds_id, files_metadata=True)
        cd = info.card_data.to_dict() if info.card_data else {}
        meta["exists"] = True
        meta["resolved_id"] = info.id
        meta["gated"] = info.gated
        meta["private"] = info.private
        meta["downloads"] = info.downloads
        meta["likes"] = info.likes
        meta["license"] = cd.get("license")
        meta["license_name"] = cd.get("license_name")
        meta["task_categories"] = cd.get("task_categories")
        meta["language"] = cd.get("language")
        meta["size_categories"] = cd.get("size_categories")
        meta["configs_in_card"] = [c.get("config_name") for c in (cd.get("configs") or [])][:20]
        di = cd.get("dataset_info")
        if di:
            dis = di if isinstance(di, list) else [di]
            meta["card_splits"] = [
                {"config": d.get("config_name"),
                 "splits": {s.get("name"): s.get("num_examples") for s in (d.get("splits") or [])},
                 "dataset_size": d.get("dataset_size")}
                for d in dis[:10]
            ]
        files = info.siblings or []
        total = sum((f.size or 0) for f in files)
        meta["n_files"] = len(files)
        meta["total_file_bytes"] = total
        meta["files_head"] = [f"{f.rfilename} ({f.size})" for f in files[:25]]
        meta["has_loading_script"] = any(f.rfilename.endswith(".py") for f in files)
    except Exception as e:  # noqa: BLE001
        meta["exists"] = False
        meta["meta_error"] = f"{type(e).__name__}: {trunc(str(e), 300)}"
    return meta


# --------------------------------------------------------------------------- loading
def try_load(ds_id: str, config: str | None, n: int) -> tuple[list[dict], str, str | None]:
    """Return (records, method, split). Raises on failure of all methods."""
    from datasets import load_dataset

    errors = []
    # 0) single data file stream (cheap; avoids resolving every shard of huge
    #    repos such as wikimedia/wikipedia, which timed out in the first run).
    if os.environ.get("PROBE_SINGLE_FILE_FIRST") == "1":
        try:
            from huggingface_hub import HfApi

            files = HfApi().list_repo_files(ds_id, repo_type="dataset")
            exts = {".parquet": "parquet", ".jsonl": "json", ".json": "json",
                    ".csv": "csv"}
            data = [f for f in files if any(f.endswith(e) for e in exts)
                    and not f.startswith(".") and "dataset_info" not in f]
            if config:
                data = [f for f in data if f.startswith(config + "/") or f"/{config}/" in f] or data
            data.sort(key=lambda f: ("train" not in f.lower(), len(f), f))
            if not data:
                raise RuntimeError("no parquet/json/csv data files")
            f0 = data[0]
            builder = next(b for e, b in exts.items() if f0.endswith(e))
            url = f"hf://datasets/{ds_id}/{f0}"
            ds = load_dataset(builder, data_files=url, split="train", streaming=True)
            recs = []
            for i, r in enumerate(ds):
                recs.append(r)
                if i + 1 >= n:
                    break
            return recs, f"single-file:{f0}", "train"
        except Exception as e:  # noqa: BLE001
            errors.append(f"single-file: {type(e).__name__}: {trunc(str(e), 300)}")
    # 1) streaming
    try:
        kw = {"streaming": True}
        ds = load_dataset(ds_id, config, **kw) if config else load_dataset(ds_id, **kw)
        split = "train" if "train" in ds else list(ds.keys())[0]
        recs = []
        for i, r in enumerate(ds[split]):
            recs.append(r)
            if i + 1 >= n:
                break
        return recs, "streaming", split
    except Exception as e:  # noqa: BLE001
        errors.append(f"streaming: {type(e).__name__}: {trunc(str(e), 300)}")
    # 2) parquet conversion branch (works for script-based datasets the Hub converted)
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        files = api.list_repo_files(ds_id, repo_type="dataset", revision="refs/convert/parquet")
        pq = [f for f in files if f.endswith(".parquet")]
        if config:
            pq = [f for f in pq if f.startswith(config + "/")] or pq
        train = [f for f in pq if "/train/" in f or "-train" in f] or pq
        if not train:
            raise RuntimeError("no parquet files on refs/convert/parquet")
        url = f"hf://datasets/{ds_id}@refs%2Fconvert%2Fparquet/{train[0]}"
        ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
        recs = []
        for i, r in enumerate(ds):
            recs.append(r)
            if i + 1 >= n:
                break
        return recs, f"parquet-convert:{train[0]}", "train"
    except Exception as e:  # noqa: BLE001
        errors.append(f"parquet-convert: {type(e).__name__}: {trunc(str(e), 300)}")
    raise RuntimeError(" | ".join(errors))


def hub_meta_retry(ds_id: str, tries: int = 3) -> dict[str, Any]:
    m: dict[str, Any] = {}
    netish = ("Connect", "SSL", "Timeout", "client has been closed")
    for i in range(tries):
        m = hub_meta(ds_id)
        err = m.get("meta_error", "")
        if m.get("exists") or not any(s in err for s in netish):
            return m
        time.sleep(3 * (i + 1))
    return m


def probe_one(ds_id: str, config: str | None, cat: str) -> dict[str, Any]:
    res: dict[str, Any] = {"id": ds_id, "config": config, "cat": cat}
    t0 = time.time()
    res["meta"] = hub_meta_retry(ds_id)
    need = N_MRC_STATS if cat == "mrc" else N_SAMPLE
    try:
        recs, method, split = try_load(ds_id, config, need)
        res["loadable"] = True
        res["method"] = method
        res["split"] = split
        first = recs[0] if recs else {}
        res["fields"] = list(first.keys())
        res["structure"] = type_tree(first)
        res["samples"] = [{k: trunc(v) for k, v in r.items()} for r in recs[:N_SAMPLE]]
        if cat == "mrc" and recs:
            ctx_keys = find_keys(first, {"context", "passage", "paragraph", "document",
                                         "documents", "context_text"})
            res["context_keys"] = ctx_keys
            res["answer_start_keys"] = find_keys(first, {"answer_start", "answer_starts",
                                                         "answer_span", "span"})
            if ctx_keys:
                lens = []
                for r in recs:
                    v = get_path(r, ctx_keys[0])
                    if isinstance(v, list):
                        v = "".join(x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
                                    for x in v)
                    if isinstance(v, str):
                        lens.append(len(v))
                if lens:
                    lens.sort()
                    res["context_len"] = {
                        "n": len(lens), "min": lens[0], "max": lens[-1],
                        "median": statistics.median(lens),
                        "mean": round(statistics.mean(lens), 1),
                        "p10": lens[len(lens) // 10], "p90": lens[(len(lens) * 9) // 10],
                    }
        if cat == "kdialog" and recs:
            res["dialog_keys"] = find_keys(first, {
                "history", "conversation", "messages", "dialog", "dialogue", "utterances",
                "knowledge", "kb", "triples", "goal", "goals", "topic", "response",
                "situation", "knowledge_graph", "attrs", "content"})
    except Exception as e:  # noqa: BLE001
        res["loadable"] = False
        res["error"] = trunc(str(e), 900)
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res


def probe_with_timeout(ds_id: str, config: str | None, cat: str, timeout: int) -> dict[str, Any]:
    spec = json.dumps({"id": ds_id, "config": config, "cat": cat})
    try:
        p = subprocess.run([sys.executable, __file__, "--probe", spec],
                           capture_output=True, text=True, timeout=timeout)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("@@RESULT@@")]
        if lines:
            return json.loads(lines[-1][len("@@RESULT@@"):])
        return {"id": ds_id, "config": config, "cat": cat, "loadable": False,
                "error": f"child exited {p.returncode}: {trunc(p.stderr[-800:], 800)}"}
    except subprocess.TimeoutExpired:
        # Child was killed -> metadata from the child is lost; fetch it here so a
        # timeout still reports existence / gating / license / size.
        return {"id": ds_id, "config": config, "cat": cat, "loadable": False,
                "error": f"TIMEOUT after {timeout}s", "meta": hub_meta_retry(ds_id)}


# --------------------------------------------------------------------------- search
def hub_search(query: str, limit: int = 15) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi

    out = []
    try:
        for d in HfApi().list_datasets(search=query, limit=limit, sort="downloads",
                                       full=True):
            cd = d.card_data.to_dict() if getattr(d, "card_data", None) else {}
            tags = d.tags or []
            lic = cd.get("license") or [t.split(":", 1)[1] for t in tags if t.startswith("license:")]
            out.append({"id": d.id, "downloads": d.downloads, "likes": d.likes,
                        "gated": d.gated, "license": lic,
                        "lang": [t for t in tags if t.startswith("language:")][:5]})
    except Exception as e:  # noqa: BLE001
        out.append({"error": f"{type(e).__name__}: {trunc(str(e), 300)}"})
    return out


# --------------------------------------------------------------------------- report
def lic_str(r: dict) -> str:
    m = r.get("meta", {})
    lic = m.get("license")
    if isinstance(lic, list):
        lic = ",".join(map(str, lic))
    return str(lic) if lic else "(none in card)"


def size_str(r: dict) -> str:
    m = r.get("meta", {})
    parts = []
    if m.get("card_splits"):
        cs = m["card_splits"][0]
        parts.append(";".join(f"{k}={v}" for k, v in cs["splits"].items()))
    if m.get("size_categories"):
        parts.append(str(m["size_categories"]))
    if m.get("total_file_bytes"):
        parts.append(f"{m['total_file_bytes'] / 1e6:.1f}MB files")
    return " ".join(parts) or "?"


def write_report(results: list[dict], searches: dict, out_dir: Path) -> str:
    L = ["# Report 034 — Phase-5 corpus source verification", "",
         f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
         "## Summary table", "",
         "| cat | dataset (config) | exists | gated | loadable | method | license | size | fields |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r.get("meta", {})
        cfg = f" ({r['config']})" if r.get("config") else ""
        L.append("| {} | `{}`{} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["cat"], r["id"], cfg, m.get("exists"), m.get("gated"),
            "YES" if r.get("loadable") else "no", r.get("method", "-"),
            lic_str(r), size_str(r).replace("|", "/"),
            ", ".join(r.get("fields", [])) or "-"))
    L += ["", "## Details", ""]
    for r in results:
        m = r.get("meta", {})
        cfg = f" / config `{r['config']}`" if r.get("config") else ""
        L += [f"### `{r['id']}`{cfg}  [{r['cat']}]", ""]
        L.append(f"- exists: {m.get('exists')}  gated: {m.get('gated')}  "
                 f"downloads: {m.get('downloads')}  license: {lic_str(r)} "
                 f"{('(' + str(m.get('license_name')) + ')') if m.get('license_name') else ''}")
        if m.get("meta_error"):
            L.append(f"- meta error: `{m['meta_error']}`")
        L.append(f"- size: {size_str(r)}  loading script: {m.get('has_loading_script')}")
        if m.get("configs_in_card"):
            L.append(f"- configs in card: {m['configs_in_card']}")
        if m.get("card_splits"):
            L.append(f"- card splits: `{json.dumps(m['card_splits'], ensure_ascii=False)[:600]}`")
        if m.get("files_head"):
            L.append(f"- files (first 25): `{', '.join(m['files_head'])[:700]}`")
        if r.get("loadable"):
            L.append(f"- **loadable** via `{r['method']}` split=`{r.get('split')}`")
            L.append("- structure:")
            L += ["```json", json.dumps(r.get("structure"), ensure_ascii=False, indent=1)[:2500], "```"]
            L.append("- sample (record 0, ≤120 chars/field):")
            for k, v in (r.get("samples") or [{}])[0].items():
                L.append(f"  - `{k}`: {v}")
            for key in ("context_keys", "answer_start_keys", "context_len", "dialog_keys"):
                if key in r:
                    L.append(f"- {key}: `{json.dumps(r[key], ensure_ascii=False)}`")
        else:
            L.append(f"- **NOT loadable**: `{r.get('error')}`")
        L.append(f"- elapsed: {r.get('elapsed_s', '?')}s")
        L.append("")
    L += ["## Hub searches (sorted by downloads)", ""]
    for cat, qs in searches.items():
        for q, hits in qs.items():
            L += [f"### [{cat}] \"{q}\"", "", "| id | downloads | gated | license | lang |",
                  "|---|---|---|---|---|"]
            for h in hits:
                if "error" in h:
                    L.append(f"| ERROR {h['error']} | | | | |")
                else:
                    L.append(f"| `{h['id']}` | {h['downloads']} | {h['gated']} | "
                             f"{h['license']} | {','.join(h['lang'])} |")
            L.append("")
    text = "\n".join(L)
    (out_dir / "report_034.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--probe", help=argparse.SUPPRESS)
    ap.add_argument("--skip-search-probe", action="store_true")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-probe failed (non-gated) entries of an existing "
                         "probe_results_034.json using single-file-first loading")
    a = ap.parse_args()

    if a.probe:
        spec = json.loads(a.probe)
        r = probe_one(spec["id"], spec["config"], spec["cat"])
        print("@@RESULT@@" + json.dumps(r, ensure_ascii=False, default=str))
        return

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if a.retry_failed:
        os.environ["PROBE_SINGLE_FILE_FIRST"] = "1"
        blob = json.loads((out / "probe_results_034.json").read_text(encoding="utf-8"))
        results, searches = blob["results"], blob["searches"]
        for i, r in enumerate(results):
            gated = (r.get("meta") or {}).get("gated")
            if r.get("loadable") or gated not in (None, False):
                continue
            print(f"[retry] {r['cat']:9s} {r['id']} {r.get('config') or ''} ...", flush=True)
            nr = probe_with_timeout(r["id"], r.get("config"), r["cat"], a.timeout)
            nr["first_attempt_error"] = r.get("error")
            print(f"        -> loadable={nr.get('loadable')} {nr.get('method', '')} "
                  f"{'' if nr.get('loadable') else trunc(nr.get('error', ''), 200)}", flush=True)
            results[i] = nr
    else:
        results, searches = run_all(a)

    finish(results, searches, out)


def run_all(a) -> tuple[list[dict], dict]:
    results: list[dict] = []
    done: set[tuple] = set()

    def run(ds_id, cfg, cat):
        if (ds_id, cfg) in done:
            return None
        done.add((ds_id, cfg))
        print(f"[probe] {cat:9s} {ds_id} {cfg or ''} ...", flush=True)
        r = probe_with_timeout(ds_id, cfg, cat, a.timeout)
        print(f"        -> loadable={r.get('loadable')} {r.get('method', '')} "
              f"{'' if r.get('loadable') else trunc(r.get('error', ''), 160)}", flush=True)
        results.append(r)
        return r

    for c in CANDIDATES:
        r = run(c["id"], c["config"], c["cat"])
        if r is not None and not r.get("loadable"):
            for alt_id, alt_cfg in c["alts"]:
                ra = run(alt_id, alt_cfg, c["cat"])
                if ra and ra.get("loadable"):
                    break

    searches: dict[str, dict[str, list]] = {}
    for cat, qs in SEARCHES.items():
        searches[cat] = {}
        for q in qs:
            print(f"[search] {cat}: {q}", flush=True)
            searches[cat][q] = hub_search(q)

    if not a.skip_search_probe:
        for cat, qs in searches.items():
            seen: list[str] = []
            for hits in qs.values():
                for h in hits:
                    if "id" in h and h["id"] not in seen:
                        seen.append(h["id"])
            # interleave queries by taking hits in download order across all queries
            ranked = sorted({h["id"]: h for hits in qs.values() for h in hits if "id" in h}.values(),
                            key=lambda h: -(h.get("downloads") or 0))
            n = 0
            for h in ranked:
                if n >= SEARCH_PROBE_TOP[cat]:
                    break
                if (h["id"], None) in done:
                    continue
                run(h["id"], None, cat)
                n += 1
    return results, searches


def finish(results: list[dict], searches: dict, out: Path) -> None:
    (out / "probe_results_034.json").write_text(
        json.dumps({"results": results, "searches": searches}, ensure_ascii=False,
                   indent=1, default=str), encoding="utf-8")
    write_report(results, searches, out)

    print("\n=== SUMMARY ===")
    print(f"{'cat':9s} {'dataset':52s} {'load':5s} {'gated':6s} {'license':22s} fields")
    for r in results:
        name = r["id"] + (f"[{r['config']}]" if r.get("config") else "")
        print(f"{r['cat']:9s} {name[:52]:52s} {'YES' if r.get('loadable') else 'no':5s} "
              f"{str(r.get('meta', {}).get('gated'))[:6]:6s} {lic_str(r)[:22]:22s} "
              f"{','.join(r.get('fields', []))[:70]}")
    print(f"\nreport: {out / 'report_034.md'}\njson:   {out / 'probe_results_034.json'}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
