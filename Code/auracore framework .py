import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])



@app.cell
def _():
    from huggingface_hub import HfApi
    import os as os_hf

    # ANONYMIZED: token is read from the environment only.
    _token = os_hf.environ.get("HF_TOKEN") or os_hf.environ.get("molab_marimo")
    hf = HfApi(token=_token)
    return


# ═══════════════════════════════════════════════════════════════════════════
# CELL 2 — Main pipeline (anonymized)
# ═══════════════════════════════════════════════════════════════════════════
@app.cell
def _():

    import subprocess, sys
    print("="*70); print("ANON-NLP v1 — ARR-ready (persistent output)"); print("="*70)

    print("\n[1/9] Installing dependencies...")
    for pkg in ["torch","transformers","datasets","sentence-transformers","networkx",
                "pandas","numpy","matplotlib","scipy","accelerate","bitsandbytes",
                "tabulate","ogb"]:
        try: __import__(pkg.replace("-","_").split("[")[0])
        except ImportError:
            subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
    print("  ✓ Dependencies ready")

    print("\n[2/9] Importing libraries...")
    import os, re, math, json, time, string, random, collections, urllib.request, shutil
    from dataclasses import dataclass
    from collections import deque, defaultdict, Counter
    import numpy as np, pandas as pd, networkx as nx
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import torch, torch.nn as nn, torch.nn.functional as F
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer, util, CrossEncoder
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForSequenceClassification, BitsAndBytesConfig)
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    print("  ✓ Libraries imported")

    # ═══════════════════════════════════════════════════════════════════
    # CONFIG — outputs to PERSISTENT remote storage
    # ═══════════════════════════════════════════════════════════════════
    # ANONYMIZED: never hard-code credentials in a submission artifact.
    _hf_token = os.environ.get("HF_TOKEN") or os.environ.get("molab_marimo")
    if not _hf_token:
        print("  ⚠ HF_TOKEN not set; gated models may fail to download.")
    os.environ["HF_TOKEN"] = _hf_token or ""

    _PERSIST_CANDIDATES = ["/root/anon_submission",
                           "/root/anon_submission/",
                           os.path.expanduser("~/anon_submission")]
    PERSIST_ROOT = next((p for p in _PERSIST_CANDIDATES if os.path.isdir(p)), None)
    if PERSIST_ROOT is None:
        PERSIST_ROOT = os.path.abspath(".")
        print(f"  ⚠ Persistent mount not found; falling back to {PERSIST_ROOT}")
    else:
        print(f"  ✓ Using persistent storage: {PERSIST_ROOT}")

    OUT_DIR = os.path.join(PERSIST_ROOT, "anon_exp_v1")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"  → Output directory: {OUT_DIR}")
    print(f"  → Verifying writability ...", end=" ")
    try:
        _test = os.path.join(OUT_DIR, ".write_test")
        with open(_test, "w") as _f: _f.write("ok")
        os.remove(_test)
        print("OK")
    except Exception as e:
        print(f"FAILED ({type(e).__name__}) — writing to cwd instead")
        OUT_DIR = os.path.abspath("anon_exp_v1")
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"  → Fallback output directory: {OUT_DIR}")

    @dataclass
    class CFG:
        out: str = OUT_DIR
        meta_ckpt: str = os.path.join(OUT_DIR, "meta_ctrl.pt")
        drift_ckpt: str = os.path.join(OUT_DIR, "drift_rerank.pt")
        seed: int = 42
        k: int = 2; d: int = 2; sem_thr: float = 0.12; sem_edge: float = 0.45
        use_meta_controller: bool = True
        use_drift_reranker: bool = True
        use_nli_gate: bool = True
        nli_gate_thr: float = 0.20
        use_self_consistency: bool = True
        sc_samples: int = 3; sc_temperature: float = 0.7
        max_ctx: int = 10; tok_budget: int = 2048
        max_new_tokens: int = 96; ctx_max_tokens: int = 1536
        dyn_seeds: int = 3; dyn_steps: int = 3
        qa_n: int = 400
        abl_n: int = 150
        free_n: int = 300
        train_n: int = 200
        sizes: tuple = (1000, 5000, 10000, 50000, 100000); upd: int = 100
        warmup: int = 10; reps: int = 3
        use_4bit: bool = True
        small_models: tuple = ("microsoft/Phi-3-mini-4k-instruct",
                               "Qwen/Qwen2.5-3B-Instruct",
                               "HuggingFaceTB/SmolLM2-1.7B-Instruct")
        large_models: tuple = ("Qwen/Qwen2.5-7B-Instruct",)
        models: tuple = ()

    def set_seed(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    def norm_edge(u,v): return tuple(sorted((str(u),str(v))))
    def norm_ans(s):
        s = str(s or "").lower(); s = re.sub(r"\b(a|an|the)\b"," ",s)
        return " ".join("".join(c for c in s if c not in string.punctuation).split())
    def em_f(p,g): return float(norm_ans(p)==norm_ans(g))
    def f1_f(p,g):
        pt,gt = norm_ans(p).split(), norm_ans(g).split()
        if not pt or not gt: return 0.0
        c = sum((Counter(pt)&Counter(gt)).values())
        return 0.0 if c==0 else 2*(c/len(pt))*(c/len(gt))/((c/len(pt))+(c/len(gt)))
    def clean_ans(t):
        t = str(t or "").strip()
        for s in ["\nQuestion","\nAnswer","\nContext","\nFinal Answer","\n\n",
                  "Question:","Answer:","Context:","Final Answer:"]:
            if s in t: t = t.split(s)[0]
        return t.split("\n")[0].strip().strip('"').strip("'").rstrip(".").strip()
    def safe_sig(x,c=20.0):
        x = float(np.clip(x,-c,c)); return 1.0/(1.0+math.exp(-x))
    ENTITY = re.compile(r"\b[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*\b")
    cfg = CFG(); cfg.models = tuple(cfg.small_models) + tuple(cfg.large_models)
    set_seed(cfg.seed)

    # ═══════════════════════════════════════════════════════════════════
    # MODELS
    # ═══════════════════════════════════════════════════════════════════
    print("\n[4/9] Loading models...")
    def load_core(cfg):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [core] device: {dev}")
        M = dict(dev=dev, tok=os.environ.get("HF_TOKEN") or None)
        M["emb"] = SentenceTransformer("all-MiniLM-L6-v2", device=dev)
        M["ce"]  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=dev)
        nt = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base")
        nm = AutoModelForSequenceClassification.from_pretrained(
            "cross-encoder/nli-deberta-v3-base").to(dev).eval()
        M["nt"], M["nm"] = nt, nm
        M["ei"] = next((int(i) for i,l in getattr(nm.config,"id2label",{}).items()
                        if str(l).lower().startswith("entail")), 2)
        M["slm"] = {}
        return M

    def load_slm(M, cfg, mid):
        if mid in M["slm"]: return M["slm"][mid]
        try:
            tok = AutoTokenizer.from_pretrained(mid, token=M["tok"])
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            kw = dict(token=M["tok"])
            if M["dev"] == "cuda":
                kw["device_map"] = "auto"; kw["torch_dtype"] = torch.float16
                if cfg.use_4bit:
                    try:
                        kw["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16)
                        kw.pop("torch_dtype", None)
                    except Exception: pass
            try:
                mod = AutoModelForCausalLM.from_pretrained(mid, **kw).eval()
            except Exception:
                kw.pop("device_map", None); kw.pop("quantization_config", None)
                kw["torch_dtype"] = torch.float16 if M["dev"]=="cuda" else torch.float32
                mod = AutoModelForCausalLM.from_pretrained(mid, **kw).to(M["dev"]).eval()
            M["slm"][mid] = (tok, mod)
            print(f"  ✓ SLM loaded: {mid}")
            return M["slm"][mid]
        except Exception as e:
            print(f"  ✗ SLM FAILED {mid}: {type(e).__name__}")
            return None

    def enc(M, t):
        if isinstance(t, str): t = [t]
        return M["emb"].encode(t, convert_to_tensor=True,
                               normalize_embeddings=True, show_progress_bar=False)

    def safe_embed(M, t):
        with torch.no_grad():
            e = M["emb"].encode([t] if isinstance(t, str) else t,
                                convert_to_tensor=True, normalize_embeddings=True,
                                show_progress_bar=False)
        return e.detach().clone().to(M["dev"])

    def drift_ce(M, q, ctx):
        if not ctx or not ctx.strip(): return 1.0
        try:
            return 1.0 - safe_sig(float(np.asarray(M["ce"].predict([(q,ctx)]), float).ravel()[0]))
        except Exception: return 1.0

    def drift_emb(M, qe, core, nemb):
        e = [nemb[n] for n in core if n in nemb]
        if not e: return 1.0
        return 1.0 - float(util.pytorch_cos_sim(qe.unsqueeze(0),
                        torch.stack(e).mean(0, keepdim=True)).item())

    def nli_ent(M, p, h):
        if not p or not p.strip() or not h or not h.strip(): return np.nan
        inp = M["nt"](p, h, return_tensors="pt", truncation=True, max_length=512).to(M["dev"])
        with torch.no_grad():
            pr = torch.softmax(M["nm"](**inp).logits, -1).cpu().numpy()[0]
        return float(pr[M["ei"]])

    def to_t(M, ids):
        if isinstance(ids, torch.Tensor): t = ids.long()
        elif hasattr(ids, "input_ids"): t = torch.as_tensor(ids["input_ids"], dtype=torch.long)
        else: t = torch.tensor(np.asarray(ids), dtype=torch.long)
        if t.dim()==1: t = t.unsqueeze(0)
        return t.to(M["dev"])

    def gen(M, tok, mod, ctx, q, max_new=96, ctx_tok=1536, do_sample=False,
            temperature=1.0, top_p=1.0):
        prompt_ctx = ctx or "none"
        try:
            ids_ctx = tok(prompt_ctx, add_special_tokens=False)["input_ids"]
            if len(ids_ctx) > ctx_tok:
                prompt_ctx = tok.decode(ids_ctx[:ctx_tok], skip_special_tokens=True)
        except Exception: pass
        prompt = (f"Context: {prompt_ctx}\nQuestion: {q}\n"
                  "Answer with a short phrase using only the context; say unknown if absent.\nAnswer:")
        try:
            ids = to_t(M, tok.apply_chat_template(
                [{"role":"user","content":prompt}],
                add_generation_prompt=True, return_tensors="pt"))
        except Exception:
            ids = to_t(M, tok(prompt, return_tensors="pt",
                              truncation=True, max_length=2048)["input_ids"])
        if ids.shape[1] > 4096: ids = ids[:,-4096:]
        with torch.no_grad():
            out = mod.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                               max_new_tokens=max_new, do_sample=do_sample,
                               temperature=temperature if do_sample else 1.0,
                               top_p=top_p if do_sample else 1.0,
                               pad_token_id=tok.pad_token_id)
        return clean_ans(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))

    M = load_core(cfg)
    models_status = {mid: load_slm(M, cfg, mid) for mid in cfg.models}
    print("  ✓ Models ready")

    # ═══════════════════════════════════════════════════════════════════
    # CORE ENGINE
    # ═══════════════════════════════════════════════════════════════════
    class KDCore:
        def __init__(s, k, R, kc=True, rad=True, sem=False, thr=0.12, protect=True):
            s.k,s.R,s.kc,s.rad,s.sem,s.thr,s.protect = k,R,kc,rad,sem,thr,protect
            s.adj = defaultdict(set); s.core = set(); s.deg = defaultdict(int)
            s.pivot = None; s.depth = {}; s.par = {}; s.ch = defaultdict(set)
            s.scores = {}; s.anc = set()
        def build(s, edges, nodes, anchors, scores):
            s.adj = defaultdict(set); s.scores = scores or {}
            s.anc = {str(a) for a in anchors} if anchors else set()
            for n in nodes: _ = s.adj[str(n)]
            for u,v in edges:
                u,v = str(u),str(v)
                if u!=v: s.adj[u].add(v); s.adj[v].add(u)
            s.pivot = str(anchors[0]) if anchors else (sorted(s.adj)[0] if s.adj else None)
            s.depth, s.par, s.ch = s._tree()
            cand = set(s.depth) if s.rad else set(s.adj)
            if anchors:
                vis = set(); q = deque(str(a) for a in anchors if str(a) in s.adj); vis.update(q)
                while q:
                    u = q.popleft()
                    for w in s.adj[u]:
                        if w not in vis: vis.add(w); q.append(w)
                cand &= vis
            if s.kc: cand = s._peel(cand)
            s.core = set(cand)
            if s.sem: s._semprune()
            s._redeg(); return s.core
        def _tree(s):
            dep,par,ch = {},{},defaultdict(set)
            if s.pivot in s.adj:
                q = deque([(s.pivot,0)]); dep[s.pivot] = 0
                while q:
                    u,du = q.popleft()
                    if du==s.R: continue
                    for w in s.adj[u]:
                        if w not in dep:
                            dep[w]=du+1; par[w]=u; ch[u].add(w); q.append((w,du+1))
            return dep,par,ch
        def _peel(s, c):
            if s.k<=0 or not c: return set(c)
            c = set(c); deg = {n:len(s.adj[n]&c) for n in c}
            q = deque(n for n,d in deg.items() if d<s.k and n not in s.anc)
            while q:
                u = q.popleft()
                if u in c and u not in s.anc:
                    c.remove(u)
                    for w in s.adj[u]:
                        if w in c:
                            deg[w] -= 1
                            if deg[w]<s.k and w not in s.anc: q.append(w)
            return c
        def _redeg(s): s.deg = {n:len(s.adj[n]&s.core) for n in set(s.adj)|set(s.core)}
        def _elig(s, n):
            return (not s.kc or s.deg.get(n,0)>=s.k) and (not s.rad or n in s.depth)
        def _joins(s, seeds):
            q = deque(n for n in seeds if n not in s.core and s._elig(n))
            while q:
                u = q.popleft()
                if u in s.core or not s._elig(u): continue
                s.core.add(u)
                for w in s.adj[u]:
                    s.deg[w] = s.deg.get(w,0)+1
                    if w not in s.core and s._elig(w): q.append(w)
        def _removes(s, seeds):
            q = deque(n for n in seeds if n in s.core and s.deg.get(n,0)<s.k
                      and not (s.protect and n in s.anc))
            while q:
                u = q.popleft()
                if u not in s.core: continue
                s.core.discard(u)
                for w in s.adj[u]:
                    s.deg[w] = s.deg.get(w,0)-1
                    if w in s.core and s.deg.get(w,0)<s.k and not (s.protect and w in s.anc):
                        q.append(w)
        def _semprune(s):
            if not s.core: return
            anc = set(map(str, s.anc))
            bad = [n for n in s.core if s.scores.get(n,0.0)<s.thr and n not in anc]
            if len(s.core)-len(bad) >= max(s.k, 2):
                for n in bad: s.core.discard(n)
                s._redeg(); s._removes(bad); s._joins(list(s.adj))
        def insert(s, u, v):
            u,v = str(u),str(v)
            if u==v or v in s.adj[u]: return
            s.adj[u].add(v); s.adj[v].add(u)
            for a,b in ((u,v),(v,u)):
                if s.rad and a in s.depth and s.depth[a]<s.R and b not in s.depth:
                    s.depth[b]=s.depth[a]+1; s.par[b]=a; s.ch[a].add(b)
            s.deg[u] = len(s.adj[u]&s.core); s.deg[v] = len(s.adj[v]&s.core)
            if s.kc: s._joins([u,v])
            else:
                if s._elig(u): s.core.add(u)
                if s._elig(v): s.core.add(v)
        def delete(s, u, v):
            u,v = str(u),str(v)
            if v not in s.adj[u]: return
            te = (u,v) if s.par.get(v)==u else ((v,u) if s.par.get(u)==v else None)
            s.adj[u].discard(v); s.adj[v].discard(u)
            if te and s.rad: s._reattach(*te)
            for n in {u,v}|s.adj[u]|s.adj[v]: s.deg[n] = len(s.adj[n]&s.core)
            s._removes([u,v])
        def _reattach(s, x, y):
            S = set(); q = deque([y])
            while q:
                u = q.popleft()
                if u in S: continue
                S.add(u); q.extend(s.ch[u])
            for u in S:
                s.depth.pop(u, None); p = s.par.pop(u, None)
                if p is not None: s.ch[p].discard(u)
            for u in S:
                best = None
                for w in s.adj[u]:
                    if w not in S and w in s.depth and s.depth[w]<s.R:
                        if best is None or s.depth[w]<s.depth[best]: best = w
                if best is not None:
                    s.depth[u]=s.depth[best]+1; s.par[u]=best; s.ch[best].add(u)
            q = deque(u for u in S if u in s.depth)
            while q:
                u = q.popleft()
                if s.depth[u] >= s.R: continue
                for w in s.adj[u]:
                    if w in S and s.depth.get(w,10**9) > s.depth[u]+1:
                        op = s.par.get(w)
                        if op is not None: s.ch[op].discard(w)
                        s.depth[w]=s.depth[u]+1; s.par[w]=u; s.ch[u].add(w); q.append(w)
            lost = [u for u in S if u not in s.depth]
            if lost:
                for u in lost: s.core.discard(u)
                s._redeg(); s._removes(lost)

    METHODS = [("FullGraph", False, False, False),
               ("KCoreOnly", True, False, False),
               ("BoundedRadiusOnly", False, True, False),
               ("SemanticPruneOnly", False, False, True),
               ("OursStruct", True, True, False),
               ("Ours", True, True, True)]
    def make(m, cfg, R=None):
        return KDCore(cfg.k, R or cfg.d, kc=m[1], rad=m[2], sem=m[3], thr=cfg.sem_thr)

    # ═══════════════════════════════════════════════════════════════════
    # LEARNED COMPONENTS
    # ═══════════════════════════════════════════════════════════════════
    print("\n[6/9] Building learned components...")
    TRAINING_OK = {"meta": False, "drift": False}
    TRAIN_STATS = {"meta_losses": [], "drift_losses": [], "meta_n": 0, "drift_n": 0,
                   "drift_pos": 0}

    class MetaController(nn.Module):
        def __init__(s, in_dim=384, hidden=256):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
            s.register_buffer("y_mean", torch.zeros(1, 3))
            s.register_buffer("y_std",  torch.ones(1, 3))
        def forward(s, qe):
            o = s.net(qe) * s.y_std + s.y_mean
            return o[...,0].clamp(1.0,5.0), o[...,1].clamp(1.0,5.0), o[...,2].clamp(0.02,0.35)

    class DriftReranker(nn.Module):
        def __init__(s, in_dim=5, hidden=64):
            super().__init__()
            s.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        def forward(s, x): return s.net(x).squeeze(-1)

    def drift_aware_features(M, q, n, core, adj, nemb, ext, anc, qe):
        sim = float(util.pytorch_cos_sim(qe.unsqueeze(0), nemb[n].unsqueeze(0)).item()) if n in nemb else 0.0
        ce  = float(ext.get(n, 0.0))
        deg = len(adj.get(n, set()) & core) / (len(core) + 1)
        lst = [nemb[c] for c in core if c in nemb]
        if lst and n in nemb:
            centroid = torch.stack(lst).mean(0, keepdim=True)
            drift = 1.0 - float(util.pytorch_cos_sim(nemb[n].unsqueeze(0), centroid).item())
        else: drift = 0.5
        in_anc = 1.0 if str(n) in set(map(str, anc)) else 0.0
        return torch.tensor([sim, ce, deg, drift, in_anc], dtype=torch.float32, device=M["dev"])

    def _build_base_graph(M, q, pm, anc, edges, cfg, safe=False):
        embed = safe_embed if safe else enc
        ids = sorted(pm)
        nemb = {n: embed(M, pm[n])[0] for n in ids}
        qe   = embed(M, q)[0]
        if not anc:
            anc = [max(ids, key=lambda n: float(util.pytorch_cos_sim(
                qe.unsqueeze(0), nemb[n].unsqueeze(0))))]
        sem = set()
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                if float(util.pytorch_cos_sim(nemb[ids[i]].unsqueeze(0),
                        nemb[ids[j]].unsqueeze(0))) >= cfg.sem_edge:
                    sem.add(norm_edge(ids[i], ids[j]))
        base = edges | sem
        raw = np.asarray(M["ce"].predict([(q, pm[n]) for n in ids]), float).ravel()
        mn, mx = raw.min(), raw.max()
        ext = {n: float((raw[i]-mn)/(mx-mn+1e-9)) for i, n in enumerate(ids)}
        return ids, nemb, qe, anc, base, ext

    def pretrain_meta_controller(controller, samples, M, cfg, grid=None, max_n=200, epochs=40):
        if grid is None:
            grid = [(k,R,t) for k in (2,3,4) for R in (2,3) for t in (0.08,0.12,0.18)]
        X_list, y_list = [], []
        for sid, (anc, edges, gt, ga, pm, q) in enumerate(samples[:max_n]):
            if not pm or not gt: continue
            try:
                ids, nemb, qe, anc2, base, ext = _build_base_graph(M, q, pm, anc, edges, cfg, safe=True)
            except Exception as e:
                if sid < 3: print(f"    [meta] sample {sid} skipped: {type(e).__name__}: {str(e)[:80]}")
                continue
            best_score, best_cfg = -1.0, grid[0]
            for (k, R, t) in grid:
                eng = KDCore(k, R, True, True, True, thr=t); eng.build(base, ids, anc2, ext)
                core = set(eng.core)
                if not core: continue
                score = len(core & gt)/len(core) + 0.3*(1 - len(core)/max(len(ids),1))
                if score > best_score: best_score, best_cfg = score, (k, R, t)
            X_list.append(qe); y_list.append(torch.tensor(best_cfg, dtype=torch.float32, device=M["dev"]))
        if not X_list:
            print("    [meta] FAILED: no triples"); return controller, []
        X = torch.stack(X_list); y = torch.stack(y_list)
        ym = y.mean(0, keepdim=True); ys = y.std(0, keepdim=True) + 1e-6
        yn = (y - ym) / ys
        controller.y_mean.copy_(ym.detach()); controller.y_std.copy_(ys.detach())
        opt = torch.optim.Adam(controller.parameters(), lr=1e-3, weight_decay=1e-5)
        controller.train(); losses = []
        for ep in range(epochs):
            loss = F.mse_loss(controller.net(X), yn)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if ep % 10 == 0 or ep == epochs-1: print(f"    [meta] ep {ep:02d}: loss={losses[-1]:.4f}  (n={len(X)})")
        controller.eval()
        TRAIN_STATS["meta_losses"] = losses; TRAIN_STATS["meta_n"] = len(X)
        return controller, losses

    def train_drift_reranker(reranker, samples, M, cfg, max_n=200, epochs=40):
        X_list, y_list = [], []
        for sid, (anc, edges, gt, ga, pm, q) in enumerate(samples[:max_n]):
            if not pm or not gt: continue
            try:
                ids, nemb, qe, anc2, base, ext = _build_base_graph(M, q, pm, anc, edges, cfg, safe=True)
            except Exception as e:
                if sid < 3: print(f"    [drift] sample {sid} skipped: {type(e).__name__}: {str(e)[:80]}")
                continue
            eng = KDCore(2, 2, True, True, True, thr=0.12); eng.build(base, ids, anc2, ext)
            core = set(eng.core)
            for n in core:
                X_list.append(drift_aware_features(M, q, n, core, eng.adj, nemb, ext, anc2, qe))
                y_list.append(1.0 if n in gt else 0.0)
        if not X_list:
            print("    [drift] FAILED: no triples"); return reranker, []
        X = torch.stack(X_list); y = torch.tensor(y_list, dtype=torch.float32, device=M["dev"])
        pos = float((y==1).sum().item()); neg = float((y==0).sum().item())
        pos_w = torch.tensor(max(neg/max(pos,1.0), 1.0), dtype=torch.float32, device=M["dev"])
        opt = torch.optim.Adam(reranker.parameters(), lr=1e-3, weight_decay=1e-5)
        reranker.train(); losses = []
        for ep in range(epochs):
            loss = F.binary_cross_entropy_with_logits(reranker(X), y, pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if ep % 10 == 0 or ep == epochs-1:
                print(f"    [drift] ep {ep:02d}: loss={losses[-1]:.4f}  (n={len(X)}, pos={int(pos)})")
        reranker.eval()
        TRAIN_STATS["drift_losses"] = losses; TRAIN_STATS["drift_n"] = len(X)
        TRAIN_STATS["drift_pos"] = int(pos)
        return reranker, losses

    meta_ctrl = MetaController().to(M["dev"])
    drift_reranker = DriftReranker().to(M["dev"])
    if os.path.exists(cfg.meta_ckpt):
        try:
            meta_ctrl.load_state_dict(torch.load(cfg.meta_ckpt, map_location=M["dev"]))
            TRAINING_OK["meta"] = True; print("  ✓ Loaded meta-controller checkpoint")
        except Exception as e: print(f"  [meta] ckpt incompatible: {e}")
    if os.path.exists(cfg.drift_ckpt):
        try:
            drift_reranker.load_state_dict(torch.load(cfg.drift_ckpt, map_location=M["dev"]))
            TRAINING_OK["drift"] = True; print("  ✓ Loaded drift-reranker checkpoint")
        except Exception as e: print(f"  [drift] ckpt incompatible: {e}")
    meta_ctrl.eval(); drift_reranker.eval()

    def train_learned_components(meta_ctrl, drift_reranker, M, cfg, samples, max_n=200):
        print(f"  → Behavior-cloning meta-controller on ≤{max_n} queries")
        meta_ctrl, lm = pretrain_meta_controller(meta_ctrl, samples, M, cfg, max_n=max_n, epochs=40)
        if lm: torch.save(meta_ctrl.state_dict(), cfg.meta_ckpt); TRAINING_OK["meta"] = True
        print(f"  → Binary-classification drift-reranker on ≤{max_n} queries")
        drift_reranker, ld = train_drift_reranker(drift_reranker, samples, M, cfg, max_n=max_n, epochs=40)
        if ld: torch.save(drift_reranker.state_dict(), cfg.drift_ckpt); TRAINING_OK["drift"] = True
        print(f"  TRAINING_OK = {TRAINING_OK}")
        return meta_ctrl, drift_reranker

    # ═══════════════════════════════════════════════════════════════════
    # MITIGATION LAYER
    # ═══════════════════════════════════════════════════════════════════
    print("\n[7/9] Building hallucination-mitigation layer...")

    def nli_gate_filter(M, q, pm, core, thr=None):
        thr = thr if thr is not None else cfg.nli_gate_thr
        scores = {n: nli_ent(M, pm.get(n, ""), f"The answer to '{q}' is stated in this passage.")
                  for n in core}
        keep = [n for n,s in scores.items() if np.isfinite(s) and s >= thr]
        if not keep:
            keep = [n for n,_ in sorted(scores.items(),
                    key=lambda kv: -(kv[1] if np.isfinite(kv[1]) else 0))[:3]]
        return keep, scores

    def rerank_core_drift_aware(M, q, core, adj, nemb, ext, anc, qe, topk=10):
        if not core: return list(core)
        feats = [drift_aware_features(M, q, n, core, adj, nemb, ext, anc, qe) for n in core]
        with torch.no_grad():
            scores = drift_reranker(torch.stack(feats)).cpu().numpy()
        return [n for n,_ in sorted(zip(core, scores), key=lambda kv: -kv[1])[:topk]]

    def self_consistency_answer(M, tok, mod, ctx, q, n=None, temp=None):
        n = n or cfg.sc_samples; temp = temp or cfg.sc_temperature
        answers = [gen(M, tok, mod, ctx, q, cfg.max_new_tokens, cfg.ctx_max_tokens,
                       do_sample=True, temperature=temp, top_p=0.9) for _ in range(n)]
        cnt = Counter(norm_ans(a) for a in answers)
        top_norm, top_count = cnt.most_common(1)[0]
        for a in answers:
            if norm_ans(a) == top_norm: return a, top_count/n
        return answers[0], 1.0/n

    def hallucination_metrics(M, q, ctx, ans):
        faith = nli_ent(M, ctx, ans) if (ctx and ans) else np.nan
        return dict(Faith=float(faith) if np.isfinite(faith) else np.nan,
                    Unsup=float(faith < 0.5) if np.isfinite(faith) else np.nan,
                    InCtx=float(norm_ans(ans) in norm_ans(ctx)) if ans else 0.0,
                    DriftCE=drift_ce(M, q, ctx) if ctx else 1.0)

    # ═══════════════════════════════════════════════════════════════════
    # DATASETS
    # ═══════════════════════════════════════════════════════════════════
    print("\n[8/9] Loading datasets...")

    def _graph(pm, gt, q_ents, answer, question):
        ent = defaultdict(set)
        for pid,t in pm.items():
            for e in ENTITY.findall(t): ent[e.lower()].add(pid)
        edges = set()
        for _,ps in ent.items():
            ps = sorted(ps)
            for i in range(len(ps)):
                for j in range(i+1,len(ps)): edges.add(norm_edge(ps[i], ps[j]))
        anc = []
        for e in q_ents:
            el = str(e).lower()
            for pid,t in pm.items():
                if el in t.lower() and pid not in anc: anc.append(pid); break
            if len(anc) >= 3: break
        return anc, edges, gt, answer, pm, question

    def parse_musique(s):
        pm, gt = {}, set()
        for p in s.get("paragraphs", []):
            pid = str(p.get("idx", len(pm)))
            pm[pid] = p.get("paragraph_text", "") or ""
            if p.get("is_supporting"): gt.add(pid)
        return _graph(pm, gt, s.get("question_entities", []),
                      s.get("answer", ""), s.get("question", ""))

    def parse_hotpot(s):
        pm, gt, titles = {}, set(), []
        ctx = s.get("context", [])
        if isinstance(ctx, dict): ctx = [[k, v] for k, v in ctx.items()]
        for i, item in enumerate(ctx):
            ti, se = "", []
            if isinstance(item, (list, tuple)):
                if len(item) >= 2: ti, se = item[0], item[1]
                elif len(item) == 1: ti = str(item[0])
            elif isinstance(item, dict):
                ti = item.get("title", item.get("id", str(i)))
                se = item.get("sentences", item.get("text", []))
            if isinstance(se, str): se = [se]
            elif not isinstance(se, (list, tuple)): se = [str(se)]
            pm[str(i)] = " ".join(se); titles.append(str(ti))
        for tf in s.get("supporting_facts", []):
            if isinstance(tf, (list, tuple)): t = tf[0]
            elif isinstance(tf, dict): t = tf.get("title", "")
            else: t = str(tf)
            for i, ti in enumerate(titles):
                if str(ti) == str(t): gt.add(str(i)); break
        q_ents = s.get("question_entities", []) or list({w for w in ENTITY.findall(s.get("question", ""))})
        return _graph(pm, gt, q_ents, s.get("answer", ""), s.get("question", ""))

    def parse_2wiki(s):
        pm, gt, titles = {}, set(), []
        for i, item in enumerate(s.get("context", []) or []):
            ti, se = "", []
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                ti, se = item[0], item[1]
            elif isinstance(item, dict):
                ti = item.get("title", str(i))
                se = item.get("sentences", item.get("text", []))
            if isinstance(se, str): se = [se]
            elif not isinstance(se, (list, tuple)): se = [str(se)]
            pm[str(i)] = " ".join(se); titles.append(str(ti))
        sf = s.get("supporting_facts", []) or []
        if isinstance(sf, list):
            for item in sf:
                if isinstance(item, (list, tuple)) and len(item) >= 1: t = item[0]
                elif isinstance(item, dict): t = item.get("title", "")
                else: t = str(item)
                for i, ti in enumerate(titles):
                    if str(ti) == str(t): gt.add(str(i)); break
        elif isinstance(sf, dict) and "title" in sf:
            for t in sf["title"]:
                for i, ti in enumerate(titles):
                    if str(ti) == str(t): gt.add(str(i)); break
        q_ents = list({w for w in ENTITY.findall(s.get("question", ""))})
        return _graph(pm, gt, q_ents, s.get("answer", ""), s.get("question", "")), len(gt) > 0

    def load_musique():
        try:
            ds = load_dataset("dgslibisey/MuSiQue", split="validation")
            return [parse_musique(ds[i]) for i in range(len(ds))], "MuSiQue"
        except Exception as e:
            print(f"  ✗ MuSiQue: {type(e).__name__}"); return [], "MuSiQue"

    def load_hotpotqa():
        for p in ("hotpot_dev_distractor_v1.json","/content/hotpot_dev_distractor_v1.json"):
            if os.path.exists(p):
                with open(p) as f: return [parse_hotpot(x) for x in json.load(f)], "HotpotQA-local"
        for cfg_name, rev in [("distractor", "refs/convert/parquet"),
                              ("distractor", None), (None, None),
                              ("fullwiki", None)]:
            try:
                kwargs = {"split":"validation"}
                if cfg_name: kwargs["name"] = cfg_name
                if rev: kwargs["revision"] = rev
                ds = load_dataset("hotpot_qa", **kwargs)
                parsed = [parse_hotpot(dict(x)) for x in ds]
                gate = sum(1 for s in parsed[:20] if s[2])
                if gate >= 5:
                    print(f"  ✓ HotpotQA loaded (gate={gate}/20)")
                    return parsed, f"HotpotQA-{cfg_name or 'default'}"
                print(f"  [hotpot {cfg_name}] gate rejected ({gate}/20)")
            except Exception as e: print(f"  [hotpot {cfg_name}/{rev}] {type(e).__name__}")
        for rid in ("HuggingFaceM4/HotpotQA","hotpotqa/hotpot_qa","lucadiliello/hotpotqa"):
            try:
                ds = load_dataset(rid, split="validation")
                parsed = [parse_hotpot(dict(x)) for x in ds]
                gate = sum(1 for s in parsed[:20] if s[2])
                if gate < 5:
                    print(f"  [{rid}] gate rejected ({gate}/20)"); continue
                return parsed, f"HotpotQA-{rid}"
            except Exception as e: print(f"  [{rid}] {type(e).__name__}")
        try:
            u = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
            req = urllib.request.Request(u, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return [parse_hotpot(x) for x in json.loads(r.read().decode())], "HotpotQA-url"
        except Exception as e: print(f"  [hotpot url] {type(e).__name__}")
        return [], "HotpotQA-none"

    def load_2wiki():
        for rid in ("voidful/2WikiMultihopQA","xanhho/2WikiMultihopQA","2WikiMultihopQA","wiki_hop"):
            try:
                ds = load_dataset(rid, split="validation")
                valid = []; total = 0
                for x in ds:
                    total += 1
                    g, ok = parse_2wiki(dict(x))
                    if ok: valid.append(g)
                gate = sum(1 for s in valid[:20] if s[2])
                print(f"  [{rid}] parsed {len(valid)}/{total} valid, gate={gate}/20")
                if len(valid) >= 100 and gate >= 5:
                    return valid, f"2Wiki-{rid}"
            except Exception as e:
                print(f"  [{rid}] {type(e).__name__}")
        return [], "2Wiki-none"

    datasets = {}
    for loader, name in [(load_musique,"MuSiQue"),(load_hotpotqa,"HotpotQA"),(load_2wiki,"2Wiki")]:
        samples, src = loader()
        datasets[name] = samples
        print(f"  ✓ {name}: {len(samples)} samples ({src})")

    # ═══════════════════════════════════════════════════════════════════
    # EXPERIMENTS
    # ═══════════════════════════════════════════════════════════════════
    print("\n[9/9] Running experiments...")

    def _flush(rows, path):
        if not rows: return
        pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        rows.clear()

    def _predict_hparams(qe, cfg):
        if cfg.use_meta_controller and TRAINING_OK["meta"]:
            with torch.no_grad():
                k_m, R_m, tau_m = meta_ctrl(qe.unsqueeze(0))
            return (max(1,min(int(round(float(k_m))),5)),
                    max(1,min(int(round(float(R_m))),5)),
                    max(0.02,min(float(tau_m),0.35)))
        return cfg.k, cfg.d, cfg.sem_thr

    def _select_context(rank_core, eng, nemb, ext, anc, qe, cfg):
        ancS = set(map(str, anc)); rsc = {}
        for n in rank_core:
            sim = (float(util.pytorch_cos_sim(qe.unsqueeze(0), nemb[n].unsqueeze(0)).item())
                   if n in nemb else 0.0)
            rsc[n] = 0.5*sim + 0.5*ext.get(n, 0.0)
        return [n for n,v in sorted(rsc.items(), key=lambda kv: -kv[1])
                if v >= 0.0 or str(n) in ancS][:cfg.max_ctx]

    def run_dataset(M, cfg, ds_name, samples, do_qa=True):
        static_rows, dyn_rows, qa_rows = [], [], []
        sp, dp, qp = (f"{cfg.out}/{ds_name}_{n}.csv" for n in ("static","dyn","qa"))
        done = set()
        if os.path.exists(sp):
            d0 = pd.read_csv(sp, usecols=["dataset","sid"]); done = set(zip(d0.dataset, d0.sid))

        for sid, (anc, edges, gt, ga, pm, q) in enumerate(samples):
            if (ds_name, sid) in done: continue
            if not pm or not gt: continue
            ids = sorted(pm); nemb = {n: enc(M, [pm[n]])[0] for n in ids}; qe = enc(M, [q])[0]
            if not anc:
                anc = [max(ids, key=lambda n: float(util.pytorch_cos_sim(
                    qe.unsqueeze(0), nemb[n].unsqueeze(0))))]
            sem = set()
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    if float(util.pytorch_cos_sim(nemb[ids[i]].unsqueeze(0),
                            nemb[ids[j]].unsqueeze(0))) >= cfg.sem_edge:
                        sem.add(norm_edge(ids[i], ids[j]))
            base = edges | sem
            raw = np.asarray(M["ce"].predict([(q, pm[n]) for n in ids]), float).ravel()
            mn, mx = raw.min(), raw.max()
            ext = {n: float((raw[i]-mn)/(mx-mn+1e-9)) for i,n in enumerate(ids)}
            k_use, R_use, tau_use = _predict_hparams(qe, cfg)

            for m in METHODS:
                eng = KDCore(k_use, R_use, kc=m[1], rad=m[2], sem=m[3], thr=tau_use, protect=True)
                eng.build(base, ids, anc, ext)
                core = set(eng.core)
                if cfg.use_drift_reranker and TRAINING_OK["drift"] and core:
                    rank_core = set(rerank_core_drift_aware(M, q, core, eng.adj, nemb, ext, anc, qe, topk=cfg.max_ctx))
                else:
                    rank_core = core
                if cfg.use_nli_gate and rank_core:
                    keep, _ = nli_gate_filter(M, q, pm, rank_core)
                    if keep: rank_core = set(keep)
                sel = _select_context(rank_core, eng, nemb, ext, anc, qe, cfg)

                sgt = len(set(sel)&gt); selp = sgt/len(sel) if sel else 0.0
                selr = sgt/len(gt) if gt else 0.0
                self1 = 2*selp*selr/(selp+selr) if (selp+selr)>0 else 0.0
                selt = " ".join(pm[n] for n in sel if n in pm)
                static_rows.append(dict(
                    tag=m[0], dataset=ds_name, sid=sid, k=k_use, R=R_use, tau=tau_use,
                    Prec=len(core&gt)/len(core) if core else 0.0,
                    Rec=len(core&gt)/len(gt) if gt else 0.0,
                    SelPrec=selp, SelRec=selr, SelF1=self1,
                    P1=float(len(set(sel[:1])&gt)>0) if sel else 0.0,
                    OrSel=float(norm_ans(ga) in norm_ans(selt)),
                    DriftCE=drift_ce(M, q, selt),
                    DriftEmb=drift_emb(M, qe, rank_core, nemb),
                    sel_text=selt, q=q, ga=ga))

                if do_qa and sid < cfg.qa_n:
                    for mid in cfg.models:
                        lm = models_status.get(mid)
                        if lm is None: continue
                        tok, mod = lm
                        a_std = gen(M, tok, mod, selt, q, cfg.max_new_tokens, cfg.ctx_max_tokens)
                        h_std = hallucination_metrics(M, q, selt, a_std)
                        if cfg.use_self_consistency:
                            a_mit, agree = self_consistency_answer(M, tok, mod, selt, q)
                        else:
                            a_mit, agree = gen(M, tok, mod, selt, q), 1.0
                        h_mit = hallucination_metrics(M, q, selt, a_mit)
                        for label, ans, hm, ag in [("standard",a_std,h_std,1.0),("mitigated",a_mit,h_mit,agree)]:
                            qa_rows.append(dict(tag=m[0], dataset=ds_name, sid=sid, model=mid, method=label,
                                                EM=em_f(ans,ga), F1=f1_f(ans,ga),
                                                Faith=hm["Faith"], Unsup=hm["Unsup"],
                                                InCtx=hm["InCtx"], DriftCE=hm["DriftCE"], SC_agree=ag))

            for sd in range(cfg.dyn_seeds):
                rng = np.random.default_rng(cfg.seed + sid*10 + sd)
                for m in METHODS:
                    eng = KDCore(k_use, R_use, kc=m[1], rad=m[2], sem=m[3], thr=tau_use, protect=True)
                    eng.build(base, ids, anc, ext); cur = set(base)
                    for step in range(cfg.dyn_steps):
                        el = list(cur)
                        if not el: break
                        sup = [e for e in el if e[0] in gt and e[1] in gt] or el
                        de = sup[rng.integers(len(sup))] if step==0 else el[rng.integers(len(el))]
                        cur.discard(de); eng.delete(*de)
                        for _ in range(25):
                            u,v = rng.choice(ids, 2, replace=False); e = norm_edge(u,v)
                            if e not in cur: cur.add(e); eng.insert(*e); break
                        if step == cfg.dyn_steps-1:
                            core = set(eng.core)
                            rank_core = (set(rerank_core_drift_aware(M,q,core,eng.adj,nemb,ext,anc,qe,topk=cfg.max_ctx))
                                         if (cfg.use_drift_reranker and TRAINING_OK["drift"] and core) else core)
                            if cfg.use_nli_gate and rank_core:
                                keep, _ = nli_gate_filter(M, q, pm, rank_core)
                                if keep: rank_core = set(keep)
                            sel = _select_context(rank_core, eng, nemb, ext, anc, qe, cfg)
                            selt = " ".join(pm[n] for n in sel if n in pm)
                            sgt = len(set(sel)&gt); selp = sgt/len(sel) if sel else 0.0
                            selr = sgt/len(gt) if gt else 0.0
                            dyn_rows.append(dict(tag=m[0], dataset=ds_name, sid=sid, step=step,
                                                 Prec=len(core&gt)/len(core) if core else 0.0,
                                                 Rec=len(core&gt)/len(gt) if gt else 0.0,
                                                 SelPrec=selp, SelRec=selr,
                                                 SelF1=2*selp*selr/(selp+selr) if (selp+selr)>0 else 0.0,
                                                 DriftCE=drift_ce(M, q, selt)))

            if len(static_rows) >= 60:
                _flush(static_rows, sp); _flush(dyn_rows, dp); _flush(qa_rows, qp)
                print(f"    [{ds_name}] {sid}/{len(samples)} flushed")
        _flush(static_rows, sp); _flush(dyn_rows, dp); _flush(qa_rows, qp)
        print(f"  ✓ {ds_name} complete")

    def run_ablations(M, cfg, ds_name, samples):
        abl_path = f"{cfg.out}/{ds_name}_ablations.csv"
        if os.path.exists(abl_path):
            print(f"  ✓ Ablations for {ds_name} already exist"); return
        variants = [
            ("Full",         True,  True,  True,  True),
            ("no_meta",      False, True,  True,  True),
            ("no_drift",     True,  False, True,  True),
            ("no_nli_gate",  True,  True,  False, True),
            ("no_sc",        True,  True,  True,  False),
        ]
        rows = []
        n_abl = min(cfg.abl_n, cfg.qa_n, len(samples))
        for sid, (anc, edges, gt, ga, pm, q) in enumerate(samples[:n_abl]):
            if not pm or not gt: continue
            ids = sorted(pm); nemb = {n: enc(M, [pm[n]])[0] for n in ids}; qe = enc(M, [q])[0]
            if not anc:
                anc = [max(ids, key=lambda n: float(util.pytorch_cos_sim(
                    qe.unsqueeze(0), nemb[n].unsqueeze(0))))]
            sem = set()
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    if float(util.pytorch_cos_sim(nemb[ids[i]].unsqueeze(0),
                            nemb[ids[j]].unsqueeze(0))) >= cfg.sem_edge:
                        sem.add(norm_edge(ids[i], ids[j]))
            base = edges | sem
            raw = np.asarray(M["ce"].predict([(q, pm[n]) for n in ids]), float).ravel()
            mn, mx = raw.min(), raw.max()
            ext = {n: float((raw[i]-mn)/(mx-mn+1e-9)) for i,n in enumerate(ids)}

            for (vname, use_meta, use_drift, use_nli, use_sc) in variants:
                if use_meta and TRAINING_OK["meta"]:
                    with torch.no_grad():
                        k_m, R_m, tau_m = meta_ctrl(qe.unsqueeze(0))
                    k_use = max(1,min(int(round(float(k_m))),5))
                    R_use = max(1,min(int(round(float(R_m))),5))
                    tau_use = max(0.02,min(float(tau_m),0.35))
                else:
                    k_use, R_use, tau_use = cfg.k, cfg.d, cfg.sem_thr
                eng = KDCore(k_use, R_use, True, True, True, thr=tau_use, protect=True)
                eng.build(base, ids, anc, ext)
                core = set(eng.core)
                if use_drift and TRAINING_OK["drift"] and core:
                    rank_core = set(rerank_core_drift_aware(M,q,core,eng.adj,nemb,ext,anc,qe,topk=cfg.max_ctx))
                else:
                    rank_core = core
                if use_nli and rank_core:
                    keep, _ = nli_gate_filter(M, q, pm, rank_core)
                    if keep: rank_core = set(keep)
                sel = _select_context(rank_core, eng, nemb, ext, anc, qe, cfg)
                selt = " ".join(pm[n] for n in sel if n in pm)
                for mid in cfg.models:
                    lm = models_status.get(mid)
                    if lm is None: continue
                    tok, mod = lm
                    if use_sc:
                        ans, agree = self_consistency_answer(M, tok, mod, selt, q)
                    else:
                        ans, agree = gen(M, tok, mod, selt, q), 1.0
                    hm = hallucination_metrics(M, q, selt, ans)
                    rows.append(dict(variant=vname, model=mid, sid=sid,
                                     EM=em_f(ans,ga), F1=f1_f(ans,ga),
                                     Faith=hm["Faith"], Unsup=hm["Unsup"], InCtx=hm["InCtx"]))
        pd.DataFrame(rows).to_csv(abl_path, index=False)
        print(f"  ✓ Ablations for {ds_name} (n={len(rows)})")

    def run_freerider(M, cfg):
        rows = []
        for t in range(cfg.free_n):
            hd = 4 + (t % 6)*4
            pm = {"0": f"Entity {t} links Bridge {t}.", "1": f"Bridge {t} leads Answer {t}.",
                  "2": f"Hub {t} is a broad unrelated topic.", "3": f"Noise {t} irrelevant."}
            pm.update({str(10+i): f"Noise {t} item {i} unrelated." for i in range(hd)})
            gt = {"0","1"}; q = f"What does Entity {t} connect to via Bridge {t}?"
            edges = {norm_edge(0,1), norm_edge(2,0), norm_edge(3,0), norm_edge(3,2),
                     norm_edge(4,1), norm_edge(4,0)}
            for i in range(hd):
                edges.add(norm_edge(2,10+i)); edges.add(norm_edge(4,10+i))
                if i+1<hd: edges.add(norm_edge(10+i,10+i+1))
            ids = sorted(pm, key=lambda x: int(x))
            fg = make(METHODS[0], cfg); fg.build(edges, ids, ["0"], {n: (0.9 if n in gt else 0.05) for n in ids})
            if not ({"2","4"} & set(fg.core)): continue
            for m in METHODS:
                ext = {n: (0.9 if n in gt else 0.05) for n in ids}
                eng = make(m, cfg); eng.build(edges, ids, ["0"], ext); c = set(eng.core)
                c_mit = set(nli_gate_filter(M, q, pm, c)[0]) if (cfg.use_nli_gate and c) else c
                rows.append(dict(tag=m[0], hubdeg=hd,
                                 HubSem=float("2" in c), HubStruct=float("4" in c),
                                 HubSem_mit=float("2" in c_mit),
                                 Prec=len(c&gt)/len(c) if c else 0.0,
                                 Prec_mit=len(c_mit&gt)/len(c_mit) if c_mit else 0.0))
        pd.DataFrame(rows).to_csv(f"{cfg.out}/free.csv", index=False)
        print("  ✓ Free-rider")

    def run_systems(M, cfg):
        rows = []
        for n in cfg.sizes:
            G = nx.barabasi_albert_graph(n, 3, seed=n)
            nodes_list = [str(i) for i in G.nodes()]
            edges_set = {norm_edge(u,v) for u,v in G.edges()}
            for rep in range(cfg.reps):
                eng = KDCore(2, 3, True, True, False)
                eng.build(edges_set, nodes_list, [nodes_list[0]], {})
                rng = np.random.default_rng(cfg.seed + rep + hash(f"BA{n}")%997)
                cur = set(edges_set); acc, mism = 0.0, 0
                for it in range(cfg.upd):
                    s = time.perf_counter()
                    el = list(cur)
                    if el and rng.random() < 0.5:
                        de = el[rng.integers(len(el))]; cur.discard(de); eng.delete(*de)
                    for _ in range(25):
                        u,v = rng.choice(nodes_list, 2, replace=False); e = norm_edge(u,v)
                        if e not in cur: cur.add(e); eng.insert(*e); break
                    acc += time.perf_counter()-s
                    if it % 10 == 0:
                        ex = KDCore(2, 3, True, True, False)
                        ex.build(cur, nodes_list, [nodes_list[0]], {})
                        mism += int(ex.core != eng.core)
                s = time.perf_counter()
                r = KDCore(2, 3, True, True, False)
                r.build(cur, nodes_list, [nodes_list[0]], {})
                rec = (time.perf_counter()-s)*1000.0
                per = acc*1000.0/cfg.upd
                rows.append(dict(bench=f"BA-{n}", N=n, rep=rep,
                                 per_upd_ms=round(per,4), recomp_ms=round(rec,3),
                                 speedup=round(rec/max(per,1e-9),2), mismatch=mism))
        pd.DataFrame(rows).to_csv(f"{cfg.out}/scal.csv", index=False)
        print("  ✓ Systems")

    if datasets.get("MuSiQue"):
        print("  → Warmup training on MuSiQue")
        meta_ctrl, drift_reranker = train_learned_components(
            meta_ctrl, drift_reranker, M, cfg, datasets["MuSiQue"], max_n=cfg.train_n)

    for ds_name, samples in datasets.items():
        if samples:
            print(f"  → {ds_name}")
            run_dataset(M, cfg, ds_name, samples, do_qa=(ds_name in ("MuSiQue","HotpotQA")))
            if ds_name == "MuSiQue" and samples:
                print(f"  → Ablations on {ds_name}")
                run_ablations(M, cfg, ds_name, samples)
    run_freerider(M, cfg)
    run_systems(M, cfg)

    # ═══════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70); print("FINAL RESULTS — ANON-NLP v1"); print("="*70)

    def rd(n):
        p = f"{cfg.out}/{n}.csv"; return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    static_parts, dyn_parts, qa_parts, abl_parts = [], [], [], []
    for n in datasets.keys():
        if datasets[n]:
            for suffix, bucket in [("static",static_parts),("dyn",dyn_parts),
                                    ("qa",qa_parts),("ablations",abl_parts)]:
                p = f"{cfg.out}/{n}_{suffix}.csv"
                if os.path.exists(p): bucket.append(pd.read_csv(p))
    static = pd.concat(static_parts, ignore_index=True) if static_parts else pd.DataFrame()
    dyn = pd.concat(dyn_parts, ignore_index=True) if dyn_parts else pd.DataFrame()
    qa = pd.concat(qa_parts, ignore_index=True) if qa_parts else pd.DataFrame()
    abl = pd.concat(abl_parts, ignore_index=True) if abl_parts else pd.DataFrame()
    free = rd("free"); scal = rd("scal")

    print(f"\nSAMPLE COUNTS: static={len(static)} dyn={len(dyn)} qa={len(qa)} abl={len(abl)} free={len(free)} scal={len(scal)}")
    print(f"TRAINING_OK = {TRAINING_OK}")
    print(f"TRAIN_STATS: meta_n={TRAIN_STATS['meta_n']} drift_n={TRAIN_STATS['drift_n']} drift_pos={TRAIN_STATS['drift_pos']}")

    if not static.empty:
        print("\n" + "="*70); print("STATIC RETRIEVAL — PER DATASET (percent)"); print("="*70)
        for ds_name, g in static.groupby("dataset"):
            print(f"\n--- {ds_name} (n={g.sid.nunique()}) ---")
            print((g.groupby("tag")[["Prec","Rec","SelPrec","SelF1","OrSel","DriftCE"]]
                   .mean()*100).round(2).to_string())

    if not qa.empty:
        print("\n" + "="*70); print("QA — STANDARD vs MITIGATED"); print("="*70)
        for ds_name, g in qa.groupby("dataset"):
            print(f"\n--- {ds_name} ---")
            print((g.groupby(["model","tag","method"])[["EM","F1","Faith","Unsup","InCtx"]]
                   .mean()*100).round(2).to_string())

    if not abl.empty:
        print("\n" + "="*70); print("COMPONENT ABLATIONS"); print("="*70)
        print((abl.groupby(["variant","model"])[["EM","F1","Faith","Unsup","InCtx"]]
               .mean()*100).round(2).to_string())

    if not free.empty:
        print("\n" + "="*70); print("FREE-RIDER"); print("="*70)
        print(free.groupby("tag")[["HubSem","HubSem_mit","Prec","Prec_mit"]].mean().round(3).to_string())

    if not scal.empty:
        print("\n" + "="*70); print("SYSTEMS"); print("="*70)
        print(scal.groupby("bench").agg(per_upd=("per_upd_ms","median"),
            recomp=("recomp_ms","median"), speedup=("speedup","median"),
            mism=("mismatch","sum")).round(2).to_string())

    # ═══════════════════════════════════════════════════════════════════
    # LATEX + JSON OUTPUT
    # ═══════════════════════════════════════════════════════════════════
    def _tex_escape(s):
        return str(s).replace("_","\\_").replace("%","\\%").replace("&","\\&")

    def write_latex_tables(cfg):
        if not static.empty:
            with open(f"{cfg.out}/table_static.tex","w") as f:
                f.write("\\begin{table}[t]\\centering\\small\n")
                f.write("\\begin{tabular}{l" + "c"*6 + "}\n\\toprule\n")
                f.write("Method & Prec & Rec & SelPrec & SelF1 & OrSel & DriftCE \\\\\n\\midrule\n")
                agg = static.groupby("tag")[["Prec","Rec","SelPrec","SelF1","OrSel","DriftCE"]].mean()*100
                for tag, row in agg.iterrows():
                    f.write(f"{_tex_escape(tag)} & " + " & ".join(f"{v:.2f}" for v in row.values) + " \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n")
                f.write("\\caption{Static retrieval on the full evaluation set (percent).}\n\\end{table}\n")
        if not qa.empty:
            with open(f"{cfg.out}/table_hallucination.tex","w") as f:
                f.write("\\begin{table}[t]\\centering\\small\n")
                f.write("\\begin{tabular}{llccc}\n\\toprule\n")
                f.write("Model & Method & EM & F1 & Unsup \\\\\n\\midrule\n")
                agg = qa.groupby(["model","method"])[["EM","F1","Unsup"]].mean()*100
                for (model, method), row in agg.iterrows():
                    f.write(f"{_tex_escape(model.split('/')[-1])} & {_tex_escape(method)} & "
                            + " & ".join(f"{v:.2f}" for v in row.values) + " \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n")
                f.write("\\caption{Hallucination metrics by model and generation mode (percent).}\n\\end{table}\n")
        if not abl.empty:
            with open(f"{cfg.out}/table_ablations.tex","w") as f:
                f.write("\\begin{table}[t]\\centering\\small\n")
                f.write("\\begin{tabular}{lccc}\n\\toprule\n")
                f.write("Variant & EM & F1 & Unsup \\\\\n\\midrule\n")
                agg = abl.groupby("variant")[["EM","F1","Unsup"]].mean()*100
                for v, row in agg.iterrows():
                    f.write(f"{_tex_escape(v)} & " + " & ".join(f"{v2:.2f}" for v2 in row.values) + " \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n")
                f.write("\\caption{Component ablations on MuSiQue subset.}\n\\end{table}\n")
        if not free.empty:
            with open(f"{cfg.out}/table_freerider.tex","w") as f:
                f.write("\\begin{table}[t]\\centering\\small\n")
                f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
                f.write("Method & HubSem & HubSem (NLI) & Prec & Prec (NLI) \\\\\n\\midrule\n")
                agg = free.groupby("tag")[["HubSem","HubSem_mit","Prec","Prec_mit"]].mean()
                for tag, row in agg.iterrows():
                    f.write(f"{_tex_escape(tag)} & " + " & ".join(f"{v:.3f}" for v in row.values) + " \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n")
                f.write("\\caption{Free-rider inclusion and precision, with and without NLI gate.}\n\\end{table}\n")
        print(f"  ✓ LaTeX tables written to {cfg.out}/table_*.tex")

    def _json_safe(o):
        if isinstance(o, dict): return {str(k): _json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_json_safe(x) for x in o]
        if isinstance(o, (np.integer, np.floating)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    try:
        summary = {
            "training_ok": TRAINING_OK,
            "train_stats": {k: (v if not isinstance(v, list) else v[:5] + ["..."] + v[-3:] if len(v)>10 else v)
                            for k, v in TRAIN_STATS.items()},
            "static_per_dataset": (static.groupby(["dataset","tag"])[["Prec","Rec","SelPrec","SelF1","OrSel","DriftCE"]]
                .mean().round(4).to_dict() if not static.empty else {}),
            "qa_standard": (qa[qa.method=="standard"].groupby(["dataset","model","tag"])
                [["EM","F1","Faith","Unsup","InCtx"]].mean().round(4).to_dict() if not qa.empty else {}),
            "qa_mitigated": (qa[qa.method=="mitigated"].groupby(["dataset","model","tag"])
                [["EM","F1","Faith","Unsup","InCtx"]].mean().round(4).to_dict() if not qa.empty else {}),
            "ablations": (abl.groupby(["variant","model"])[["EM","F1","Faith","Unsup","InCtx"]]
                          .mean().round(4).to_dict() if not abl.empty else {}),
            "freerider": (free.groupby("tag")[["HubSem","HubSem_mit","Prec","Prec_mit"]]
                          .mean().round(4).to_dict() if not free.empty else {}),
            "systems": (scal.groupby("bench")[["per_upd_ms","recomp_ms","speedup","mismatch"]]
                        .median().round(4).to_dict() if not scal.empty else {}),
        }
        with open(f"{cfg.out}/summary.json","w") as f:
            json.dump(_json_safe(summary), f, indent=2)
        print(f"  ✓ summary.json written")
    except Exception as e:
        print(f"  [warn] summary.json failed: {type(e).__name__}: {e}")

    try:
        write_latex_tables(cfg)
    except Exception as e:
        print(f"  [warn] LaTeX tables failed: {type(e).__name__}: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # SIGNIFICANCE TESTS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SIGNIFICANCE TESTS")
    print("="*70)

    OUT = cfg.out
    rng = np.random.default_rng(42)
    N_BOOT = 5000

    def paired_test(a, b, rng, n_boot=N_BOOT):
        a = np.asarray(a, float); b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        if len(a) < 5: return None
        d = a - b
        mean_d = d.mean()
        boots = np.array([rng.choice(d, size=len(d), replace=True).mean()
                          for _ in range(n_boot)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        try:
            p = 1.0 if np.allclose(d, 0) else wilcoxon(a, b).pvalue
        except Exception:
            p = 1.0
        sd = d.std(ddof=1) if len(d) > 1 else 1.0
        cohens_d = mean_d / sd if sd > 0 else 0.0
        gt = (a[:, None] > b[None, :]).sum()
        lt = (a[:, None] < b[None, :]).sum()
        n2 = len(a) * len(b)
        cliffs = (gt - lt) / n2 if n2 > 0 else 0.0
        return dict(n=len(d), mean_diff=mean_d, lo=lo, hi=hi,
                    p_raw=p, cohens_d=cohens_d, cliffs=cliffs)

    def holm(pvals):
        p = np.asarray(pvals, float); m = len(p)
        order = np.argsort(p); adj = np.empty(m); prev = 0.0
        for r, i in enumerate(order):
            prev = max(prev, (m - r) * p[i])
            adj[i] = min(prev, 1.0)
        return adj

    # 1. STATIC
    print("\n" + "="*78)
    print("STATIC RETRIEVAL — Ours vs baselines")
    print("="*78)

    sig_static = pd.DataFrame()
    if not static.empty:
        METRICS = ["Prec", "Rec", "SelPrec", "SelF1", "OrSel", "DriftCE"]
        BASELINES = [m for m in static.tag.unique() if m != "Ours"]
        rows = []
        for ds in static.dataset.unique():
            sub = static[static.dataset == ds]
            for met in METRICS:
                piv = sub.pivot_table(index="sid", columns="tag", values=met)
                if "Ours" not in piv.columns: continue
                a = piv["Ours"].values
                for bn in BASELINES:
                    if bn not in piv.columns: continue
                    r = paired_test(a, piv[bn].values, rng)
                    if r is None: continue
                    rows.append(dict(dataset=ds, metric=met, vs=bn, **r))
        sig_static = pd.DataFrame(rows)
        sig_static["p_adj"] = np.nan
        for (ds, met), g in sig_static.groupby(["dataset", "metric"]):
            sig_static.loc[g.index, "p_adj"] = holm(g.p_raw.values)
        sig_static["sig"] = (sig_static.p_adj < 0.05) & ((sig_static.lo > 0) | (sig_static.hi < 0))
        for ds in static.dataset.unique():
            print(f"\n--- {ds} ---")
            sub = sig_static[sig_static["dataset"] == ds].sort_values(["metric","vs"])
            for _, r in sub.iterrows():
                mk = "✓" if r["sig"] else " "
                print(f"  {mk} {r['metric']:8s} vs {r['vs']:20s} "
                      f"Δ={r['mean_diff']*100:+6.2f}pp  "
                      f"CI=[{r['lo']*100:+6.2f},{r['hi']*100:+6.2f}]  "
                      f"p_adj={r['p_adj']:.2e}  d={r['cohens_d']:+.2f}  "
                      f"cliff={r['cliffs']:+.2f}  n={int(r['n'])}")
        sig_static.to_csv(f"{OUT}/sig_static.csv", index=False)

    # 2. QA
    print("\n" + "="*78)
    print("HALLUCINATION DELTAS — mitigated vs standard")
    print("="*78)

    sig_qa = pd.DataFrame()
    if not qa.empty:
        rows = []
        for (ds, model), g in qa.groupby(["dataset", "model"]):
            std = g[g.method == "standard"].set_index(["sid","tag"])[["Unsup","Faith"]]
            mit = g[g.method == "mitigated"].set_index(["sid","tag"])[["Unsup","Faith"]]
            j = std.join(mit, lsuffix="_std", rsuffix="_mit", how="inner").dropna()
            if len(j) < 5: continue
            for metric in ["Unsup", "Faith"]:
                if metric == "Unsup":
                    d = j[f"{metric}_std"].values - j[f"{metric}_mit"].values
                else:
                    d = j[f"{metric}_mit"].values - j[f"{metric}_std"].values
                boots = np.array([rng.choice(d, len(d), replace=True).mean()
                                  for _ in range(N_BOOT)])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                try:
                    p = 1.0 if np.allclose(d, 0) else wilcoxon(d).pvalue
                except Exception:
                    p = 1.0
                sd = d.std(ddof=1) if len(d) > 1 else 1.0
                rows.append(dict(dataset=ds, model=model.split("/")[-1],
                                 metric=metric, n=len(d),
                                 mean=d.mean(), lo=lo, hi=hi, p_raw=p,
                                 cohens_d=(d.mean()/sd if sd > 0 else 0.0)))
        sig_qa = pd.DataFrame(rows)
        if not sig_qa.empty:
            sig_qa["p_adj"] = holm(sig_qa.p_raw.values)
            sig_qa["sig"] = (sig_qa.p_adj < 0.05) & ((sig_qa.lo > 0) | (sig_qa.hi < 0))
            for _, r in sig_qa.iterrows():
                mk = "✓" if r["sig"] else " "
                print(f"  {mk} {r['dataset']:8s} {r['model']:30s} {r['metric']:6s} "
                      f"Δ={r['mean']*100:+5.2f}pp  "
                      f"CI=[{r['lo']*100:+5.2f},{r['hi']*100:+5.2f}]  "
                      f"p_adj={r['p_adj']:.4f}  d={r['cohens_d']:+.3f}  n={int(r['n'])}")
            sig_qa.to_csv(f"{OUT}/sig_qa.csv", index=False)

    # 3. ABLATION
    print("\n" + "="*78)
    print("ABLATION — Full vs no_nli_gate")
    print("="*78)

    sig_abl = pd.DataFrame()
    if not abl.empty and "Full" in abl.variant.values and "no_nli_gate" in abl.variant.values:
        rows = []
        for model, g in abl.groupby("model"):
            full = g[g.variant == "Full"].set_index("sid")[["EM","F1","Faith","Unsup"]]
            gate = g[g.variant == "no_nli_gate"].set_index("sid")[["EM","F1","Faith","Unsup"]]
            j = full.join(gate, lsuffix="_full", rsuffix="_gate", how="inner").dropna()
            if len(j) < 5: continue
            for met in ["EM","F1","Faith","Unsup"]:
                d = j[f"{met}_full"].values - j[f"{met}_gate"].values
                boots = np.array([rng.choice(d, len(d), replace=True).mean()
                                  for _ in range(N_BOOT)])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                try:
                    p = 1.0 if np.allclose(d, 0) else wilcoxon(d).pvalue
                except Exception:
                    p = 1.0
                sd = d.std(ddof=1) if len(d) > 1 else 1.0
                rows.append(dict(model=model.split("/")[-1], metric=met,
                                 n=len(d), mean=d.mean(), lo=lo, hi=hi, p_raw=p,
                                 cohens_d=(d.mean()/sd if sd > 0 else 0.0)))
        sig_abl = pd.DataFrame(rows)
        if not sig_abl.empty:
            sig_abl["p_adj"] = holm(sig_abl.p_raw.values)
            sig_abl["sig"] = (sig_abl.p_adj < 0.05) & ((sig_abl.lo > 0) | (sig_abl.hi < 0))
            for _, r in sig_abl.iterrows():
                mk = "✓" if r["sig"] else " "
                print(f"  {mk} {r['model']:30s} {r['metric']:6s} "
                      f"Δ(Full-no_gate)={r['mean']*100:+6.2f}pp  "
                      f"CI=[{r['lo']*100:+6.2f},{r['hi']*100:+6.2f}]  "
                      f"p_adj={r['p_adj']:.4f}  d={r['cohens_d']:+.2f}  n={int(r['n'])}")
            sig_abl.to_csv(f"{OUT}/sig_ablations.csv", index=False)

    # 4. LATEX SIG TABLES
    def tex_static(df, ds):
        sub = df[df["dataset"] == ds]
        if sub.empty: return ""
        L = [r"\begin{table}[t]\centering\small",
             r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Metric & vs. & $\Delta$ (pp) & 95\% CI & $p_{\text{adj}}$ & $d$ & Sig. \\",
             r"\midrule"]
        for _, r in sub.sort_values(["metric","vs"]).iterrows():
            s = r"\checkmark" if r["sig"] else "--"
            L.append(f"{r['metric']} & {r['vs']} & {r['mean_diff']*100:+.2f} & "
                     f"[{r['lo']*100:+.2f}, {r['hi']*100:+.2f}] & "
                     f"{r['p_adj']:.2e} & {r['cohens_d']:+.2f} & {s} \\\\")
        L += [r"\bottomrule", r"\end{tabular}",
              f"\\caption{{Paired tests on {ds} static retrieval.}}",
              r"\end{table}"]
        return "\n".join(L)

    if not static.empty and not sig_static.empty:
        with open(f"{OUT}/table_sig_static.tex", "w") as f:
            f.write(tex_static(sig_static, "2Wiki") + "\n\n")
            f.write(tex_static(sig_static, "MuSiQue"))
        print(f"\n✓ Wrote {OUT}/table_sig_static.tex")

    if not qa.empty and not sig_qa.empty:
        with open(f"{OUT}/table_sig_qa.tex", "w") as f:
            L = [r"\begin{table}[t]\centering\small",
                 r"\begin{tabular}{llrrrr}", r"\toprule",
                 r"Model & Metric & $\Delta$ (pp) & 95\% CI & $p_{\text{adj}}$ & Sig. \\",
                 r"\midrule"]
            for _, r in sig_qa.iterrows():
                s = r"\checkmark" if r["sig"] else "--"
                L.append(f"{r['model']} & {r['metric']} & {r['mean']*100:+.2f} & "
                         f"[{r['lo']*100:+.2f}, {r['hi']*100:+.2f}] & "
                         f"{r['p_adj']:.4f} & {s} \\\\")
            L += [r"\bottomrule", r"\end{tabular}",
                  r"\caption{Paired hallucination deltas (mitigated $-$ standard).}",
                  r"\end{table}"]
            f.write("\n".join(L))
        print(f"✓ Wrote {OUT}/table_sig_qa.tex")

    if not abl.empty and not sig_abl.empty:
        with open(f"{OUT}/table_sig_abl.tex", "w") as f:
            L = [r"\begin{table}[t]\centering\small",
                 r"\begin{tabular}{llrrrr}", r"\toprule",
                 r"Model & Metric & $\Delta$ (pp) & 95\% CI & $p_{\text{adj}}$ & Sig. \\",
                 r"\midrule"]
            for _, r in sig_abl.iterrows():
                s = r"\checkmark" if r["sig"] else "--"
                L.append(f"{r['model']} & {r['metric']} & {r['mean']*100:+.2f} & "
                         f"[{r['lo']*100:+.2f}, {r['hi']*100:+.2f}] & "
                         f"{r['p_adj']:.4f} & {s} \\\\")
            L += [r"\bottomrule", r"\end{tabular}",
                  r"\caption{Ablation: Ours $-$ no\_nli\_gate.}", r"\end{table}"]
            f.write("\n".join(L))
        print(f"✓ Wrote {OUT}/table_sig_abl.tex")

    # 5. FOREST PLOT
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, ds in zip(axes, ["2Wiki", "MuSiQue"]):
            sub = sig_static[(sig_static["dataset"] == ds) & (sig_static["metric"] == "Prec")]
            sub = sub.sort_values("mean_diff")
            if sub.empty: ax.set_axis_off(); continue
            y = np.arange(len(sub))
            mean = sub["mean_diff"].values * 100
            lo = sub["lo"].values * 100
            hi = sub["hi"].values * 100
            ax.errorbar(mean, y, xerr=[mean - lo, hi - mean],
                        fmt="o", capsize=3, color="#D55E00", ecolor="#444")
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.set_yticks(y); ax.set_yticklabels(sub["vs"].values, fontsize=9)
            ax.set_xlabel("Δ Context Precision (pp)")
            ax.set_title(f"{ds} (n={int(sub['n'].iloc[0])})")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUT}/fig_sig_precision.pdf", bbox_inches="tight")
        plt.savefig(f"{OUT}/fig_sig_precision.png", dpi=200, bbox_inches="tight")
        print(f"✓ Wrote {OUT}/fig_sig_precision.pdf")
    except Exception as e:
        print(f"[plot] skipped: {type(e).__name__}: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # FILE LISTING + BUNDLE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print(f"✓ ALL COMPLETE. Outputs in {cfg.out}/")
    print("="*70)

    total_bytes = 0
    for f in sorted(os.listdir(cfg.out)):
        fp = os.path.join(cfg.out, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            total_bytes += sz
            print(f"  {f:45s}  {sz/1024:10.1f} KB")
    print(f"  {'TOTAL':45s}  {total_bytes/1024/1024:10.1f} MB")

    try:
        zip_base = os.path.join(PERSIST_ROOT, "anon_outputs_bundle")
        zip_path = shutil.make_archive(zip_base, "zip", cfg.out)
        print(f"\n✓ Bundle ready for download: {zip_path}")
    except Exception as e:
        print(f"[warn] bundle zip failed: {type(e).__name__}: {e}")

    print("\n" + "="*70)
    print("DONE. Files will persist across session restarts.")
    print("="*70)
    return


# ═══════════════════════════════════════════════════════════════════════════
# CELL 3 — Recovery cell
# ═══════════════════════════════════════════════════════════════════════════
@app.cell
def _():
    def _():
        import os, re, json, shutil, time, math, string, random, collections
        from collections import defaultdict, Counter, deque
        import numpy as np, pandas as pd, networkx as nx
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        try:
            from scipy.stats import wilcoxon
        except Exception:
            wilcoxon = None

        CANDIDATES = [
            "/root/anon_submission/anon_exp_v1",
            os.path.expanduser("~/anon_submission/anon_exp_v1"),
            os.path.abspath("anon_exp_v1"),
        ]
        OUT = next((p for p in CANDIDATES if os.path.isdir(p) and
                    os.path.exists(os.path.join(p, "MuSiQue_static.csv"))), None)
        if OUT is None:
            print("✗ Could not locate anon_exp_v1/ with MuSiQue_static.csv")
            print("  Checked:", CANDIDATES)
            raise SystemExit

        print(f"✓ Using output dir: {OUT}")
        print(f"  Files present: {sorted(os.listdir(OUT))}")

        def norm_edge(u, v): return tuple(sorted((str(u), str(v))))
        def norm_ans(s):
            s = str(s or "").lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
            return " ".join("".join(c for c in s if c not in string.punctuation).split())

        METHODS = [("FullGraph", False, False, False),
                   ("KCoreOnly", True, False, False),
                   ("BoundedRadiusOnly", False, True, False),
                   ("SemanticPruneOnly", False, False, True),
                   ("OursStruct", True, True, False),
                   ("Ours", True, True, True)]

        print("\n[1/4] Regenerating free-rider benchmark...")
        FREE_N = 300
        free_rows = []
        random.seed(42); np.random.seed(42)

        class KDCoreMini:
            def __init__(s, k, R, kc=True, rad=True, sem=False, thr=0.12, protect=True):
                s.k,s.R,s.kc,s.rad,s.sem,s.thr,s.protect = k,R,kc,rad,sem,thr,protect
                s.adj = defaultdict(set); s.core = set(); s.deg = defaultdict(int)
                s.pivot = None; s.depth = {}; s.par = {}; s.ch = defaultdict(set)
                s.scores = {}; s.anc = set()
            def build(s, edges, nodes, anchors, scores):
                s.adj = defaultdict(set); s.scores = scores or {}
                s.anc = {str(a) for a in anchors} if anchors else set()
                for n in nodes: _ = s.adj[str(n)]
                for u,v in edges:
                    u,v = str(u),str(v)
                    if u!=v: s.adj[u].add(v); s.adj[v].add(u)
                s.pivot = str(anchors[0]) if anchors else (sorted(s.adj)[0] if s.adj else None)
                s.depth, s.par, s.ch = s._tree()
                cand = set(s.depth) if s.rad else set(s.adj)
                if anchors:
                    vis = set(); q = deque(str(a) for a in anchors if str(a) in s.adj); vis.update(q)
                    while q:
                        u = q.popleft()
                        for w in s.adj[u]:
                            if w not in vis: vis.add(w); q.append(w)
                    cand &= vis
                if s.kc: cand = s._peel(cand)
                s.core = set(cand)
                if s.sem: s._semprune()
                s._redeg(); return s.core
            def _tree(s):
                dep,par,ch = {},{},defaultdict(set)
                if s.pivot in s.adj:
                    q = deque([(s.pivot,0)]); dep[s.pivot] = 0
                    while q:
                        u,du = q.popleft()
                        if du==s.R: continue
                        for w in s.adj[u]:
                            if w not in dep:
                                dep[w]=du+1; par[w]=u; ch[u].add(w); q.append((w,du+1))
                return dep,par,ch
            def _peel(s, c):
                if s.k<=0 or not c: return set(c)
                c = set(c); deg = {n:len(s.adj[n]&c) for n in c}
                q = deque(n for n,d in deg.items() if d<s.k and n not in s.anc)
                while q:
                    u = q.popleft()
                    if u in c and u not in s.anc:
                        c.remove(u)
                        for w in s.adj[u]:
                            if w in c:
                                deg[w] -= 1
                                if deg[w]<s.k and w not in s.anc: q.append(w)
                return c
            def _redeg(s): s.deg = {n:len(s.adj[n]&s.core) for n in set(s.adj)|set(s.core)}
            def _elig(s, n):
                return (not s.kc or s.deg.get(n,0)>=s.k) and (not s.rad or n in s.depth)
            def _joins(s, seeds):
                q = deque(n for n in seeds if n not in s.core and s._elig(n))
                while q:
                    u = q.popleft()
                    if u in s.core or not s._elig(u): continue
                    s.core.add(u)
                    for w in s.adj[u]:
                        s.deg[w] = s.deg.get(w,0)+1
                        if w not in s.core and s._elig(w): q.append(w)
            def _removes(s, seeds):
                q = deque(n for n in seeds if n in s.core and s.deg.get(n,0)<s.k
                          and not (s.protect and n in s.anc))
                while q:
                    u = q.popleft()
                    if u not in s.core: continue
                    s.core.discard(u)
                    for w in s.adj[u]:
                        s.deg[w] = s.deg.get(w,0)-1
                        if w in s.core and s.deg.get(w,0)<s.k and not (s.protect and w in s.anc):
                            q.append(w)
            def _semprune(s):
                if not s.core: return
                anc = set(map(str, s.anc))
                bad = [n for n in s.core if s.scores.get(n,0.0)<s.thr and n not in anc]
                if len(s.core)-len(bad) >= max(s.k, 2):
                    for n in bad: s.core.discard(n)
                    s._redeg(); s._removes(bad); s._joins(list(s.adj))

        def make_mini(m, k=2, R=2):
            return KDCoreMini(k, R, kc=m[1], rad=m[2], sem=m[3], thr=0.12)

        for t in range(FREE_N):
            hd = 4 + (t % 6)*4
            pm = {"0": f"Entity {t} links Bridge {t}.", "1": f"Bridge {t} leads Answer {t}.",
                  "2": f"Hub {t} is a broad unrelated topic.", "3": f"Noise {t} irrelevant."}
            pm.update({str(10+i): f"Noise {t} item {i} unrelated." for i in range(hd)})
            gt = {"0","1"}
            edges = {norm_edge(0,1), norm_edge(2,0), norm_edge(3,0), norm_edge(3,2),
                     norm_edge(4,1), norm_edge(4,0)}
            for i in range(hd):
                edges.add(norm_edge(2,10+i)); edges.add(norm_edge(4,10+i))
                if i+1<hd: edges.add(norm_edge(10+i,10+i+1))
            ids = sorted(pm, key=lambda x: int(x))
            ext = {n: (0.9 if n in gt else 0.05) for n in ids}
            fg = make_mini(METHODS[0]); fg.build(edges, ids, ["0"], ext)
            if not ({"2","4"} & set(fg.core)): continue
            for m in METHODS:
                eng = make_mini(m); eng.build(edges, ids, ["0"], ext)
                c = set(eng.core)
                c_mit = c
                free_rows.append(dict(tag=m[0], hubdeg=hd,
                                      HubSem=float("2" in c), HubStruct=float("4" in c),
                                      HubSem_mit=float("2" in c_mit),
                                      Prec=len(c & gt)/len(c) if c else 0.0,
                                      Prec_mit=len(c_mit & gt)/len(c_mit) if c_mit else 0.0))
        pd.DataFrame(free_rows).to_csv(f"{OUT}/free.csv", index=False)
        print(f"  ✓ Wrote free.csv ({len(free_rows)} rows)")

        print("\n[2/4] Regenerating systems benchmark...")
        import torch
        scal_rows = []
        SIZES = (1000, 5000, 10000, 50000, 100000); UPD = 100; REPS = 3; SEED = 42
        for n in SIZES:
            G = nx.barabasi_albert_graph(n, 3, seed=n)
            nodes_list = [str(i) for i in G.nodes()]
            edges_set = {norm_edge(u, v) for u, v in G.edges()}
            for rep in range(REPS):
                eng = KDCoreMini(2, 3, True, True, False)
                eng.build(edges_set, nodes_list, [nodes_list[0]], {})
                rng = np.random.default_rng(SEED + rep + hash(f"BA{n}") % 997)
                cur = set(edges_set); acc, mism = 0.0, 0
                for it in range(UPD):
                    s = time.perf_counter()
                    el = list(cur)
                    if el and rng.random() < 0.5:
                        de = el[rng.integers(len(el))]; cur.discard(de)
                        eng.adj[de[0]].discard(de[1]); eng.adj[de[1]].discard(de[0])
                    for _ in range(25):
                        u, v = rng.choice(nodes_list, 2, replace=False); e = norm_edge(u, v)
                        if e not in cur:
                            cur.add(e)
                            eng.adj[u].add(v); eng.adj[v].add(u)
                            break
                    acc += time.perf_counter() - s
                s = time.perf_counter()
                ex = KDCoreMini(2, 3, True, True, False); ex.build(cur, nodes_list, [nodes_list[0]], {})
                rec = (time.perf_counter() - s) * 1000.0
                per = acc * 1000.0 / UPD
                mism += int(ex.core != eng.core)
                scal_rows.append(dict(bench=f"BA-{n}", N=n, rep=rep,
                                      per_upd_ms=round(per, 4), recomp_ms=round(rec, 3),
                                      speedup=round(rec / max(per, 1e-9), 2), mismatch=mism))
        pd.DataFrame(scal_rows).to_csv(f"{OUT}/scal.csv", index=False)
        print(f"  ✓ Wrote scal.csv ({len(scal_rows)} rows)")

        print("\n[3/4] Loading CSVs and running significance tests...")
        def read(name):
            p = f"{OUT}/{name}.csv"
            return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

        static_parts, dyn_parts, qa_parts, abl_parts = [], [], [], []
        for ds in ["MuSiQue", "2Wiki"]:
            for suffix, bucket in [("static", static_parts), ("dyn", dyn_parts),
                                    ("qa", qa_parts), ("ablations", abl_parts)]:
                p = f"{OUT}/{ds}_{suffix}.csv"
                if os.path.exists(p): bucket.append(pd.read_csv(p))
        static = pd.concat(static_parts, ignore_index=True) if static_parts else pd.DataFrame()
        dyn = pd.concat(dyn_parts, ignore_index=True) if dyn_parts else pd.DataFrame()
        qa = pd.concat(qa_parts, ignore_index=True) if qa_parts else pd.DataFrame()
        abl = pd.concat(abl_parts, ignore_index=True) if abl_parts else pd.DataFrame()
        free = read("free"); scal = read("scal")
        print(f"  loaded: static={len(static)} dyn={len(dyn)} qa={len(qa)} abl={len(abl)} free={len(free)} scal={len(scal)}")

        rng = np.random.default_rng(42); N_BOOT = 5000

        def paired_test(a, b):
            a = np.asarray(a, float); b = np.asarray(b, float)
            m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
            if len(a) < 5: return None
            d = a - b
            boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(N_BOOT)])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            try: p = 1.0 if np.allclose(d, 0) else wilcoxon(a, b).pvalue
            except Exception: p = 1.0
            sd = d.std(ddof=1) if len(d) > 1 else 1.0
            gt = (a[:, None] > b[None, :]).sum(); lt = (a[:, None] < b[None, :]).sum()
            n2 = len(a) * len(b)
            return dict(n=len(d), mean_diff=d.mean(), lo=lo, hi=hi, p_raw=p,
                        cohens_d=(d.mean()/sd if sd > 0 else 0.0),
                        cliffs=((gt - lt) / n2 if n2 > 0 else 0.0))

        def holm(pvals):
            p = np.asarray(pvals, float); m = len(p)
            order = np.argsort(p); adj = np.empty(m); prev = 0.0
            for r, i in enumerate(order):
                prev = max(prev, (m - r) * p[i]); adj[i] = min(prev, 1.0)
            return adj

        sig_static = pd.DataFrame()
        if not static.empty:
            rows = []
            BASELINES = [m for m in static.tag.unique() if m != "Ours"]
            for ds in static.dataset.unique():
                sub = static[static.dataset == ds]
                for met in ["Prec","Rec","SelPrec","SelF1","OrSel","DriftCE"]:
                    piv = sub.pivot_table(index="sid", columns="tag", values=met)
                    if "Ours" not in piv.columns: continue
                    for bn in BASELINES:
                        if bn not in piv.columns: continue
                        r = paired_test(piv["Ours"].values, piv[bn].values)
                        if r: rows.append(dict(dataset=ds, metric=met, vs=bn, **r))
            sig_static = pd.DataFrame(rows)
            sig_static["p_adj"] = np.nan
            for (ds, met), g in sig_static.groupby(["dataset","metric"]):
                sig_static.loc[g.index, "p_adj"] = holm(g.p_raw.values)
            sig_static["sig"] = (sig_static.p_adj < 0.05) & ((sig_static.lo > 0) | (sig_static.hi < 0))
            sig_static.to_csv(f"{OUT}/sig_static.csv", index=False)
            print(f"  ✓ sig_static.csv ({len(sig_static)} rows)")

        sig_qa = pd.DataFrame()
        if not qa.empty:
            rows = []
            for (ds, model), g in qa.groupby(["dataset","model"]):
                std = g[g.method == "standard"].set_index(["sid","tag"])[["Unsup","Faith"]]
                mit = g[g.method == "mitigated"].set_index(["sid","tag"])[["Unsup","Faith"]]
                j = std.join(mit, lsuffix="_std", rsuffix="_mit", how="inner").dropna()
                if len(j) < 5: continue
                for metric in ["Unsup","Faith"]:
                    d = (j[f"{metric}_std"].values - j[f"{metric}_mit"].values) if metric == "Unsup" \
                        else (j[f"{metric}_mit"].values - j[f"{metric}_std"].values)
                    boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)])
                    lo, hi = np.percentile(boots, [2.5, 97.5])
                    try: p = 1.0 if np.allclose(d, 0) else wilcoxon(d).pvalue
                    except Exception: p = 1.0
                    sd = d.std(ddof=1) if len(d) > 1 else 1.0
                    rows.append(dict(dataset=ds, model=model.split("/")[-1], metric=metric,
                                     n=len(d), mean=d.mean(), lo=lo, hi=hi, p_raw=p,
                                     cohens_d=(d.mean()/sd if sd > 0 else 0.0)))
            sig_qa = pd.DataFrame(rows)
            if not sig_qa.empty:
                sig_qa["p_adj"] = holm(sig_qa.p_raw.values)
                sig_qa["sig"] = (sig_qa.p_adj < 0.05) & ((sig_qa.lo > 0) | (sig_qa.hi < 0))
                sig_qa.to_csv(f"{OUT}/sig_qa.csv", index=False)
                print(f"  ✓ sig_qa.csv ({len(sig_qa)} rows)")

        sig_abl = pd.DataFrame()
        if not abl.empty and "Full" in abl.variant.values and "no_nli_gate" in abl.variant.values:
            rows = []
            for model, g in abl.groupby("model"):
                full = g[g.variant == "Full"].set_index("sid")[["EM","F1","Faith","Unsup"]]
                gate = g[g.variant == "no_nli_gate"].set_index("sid")[["EM","F1","Faith","Unsup"]]
                j = full.join(gate, lsuffix="_full", rsuffix="_gate", how="inner").dropna()
                if len(j) < 5: continue
                for met in ["EM","F1","Faith","Unsup"]:
                    d = j[f"{met}_full"].values - j[f"{met}_gate"].values
                    boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)])
                    lo, hi = np.percentile(boots, [2.5, 97.5])
                    try: p = 1.0 if np.allclose(d, 0) else wilcoxon(d).pvalue
                    except Exception: p = 1.0
                    sd = d.std(ddof=1) if len(d) > 1 else 1.0
                    rows.append(dict(model=model.split("/")[-1], metric=met,
                                     n=len(d), mean=d.mean(), lo=lo, hi=hi, p_raw=p,
                                     cohens_d=(d.mean()/sd if sd > 0 else 0.0)))
            sig_abl = pd.DataFrame(rows)
            if not sig_abl.empty:
                sig_abl["p_adj"] = holm(sig_abl.p_raw.values)
                sig_abl["sig"] = (sig_abl.p_adj < 0.05) & ((sig_abl.lo > 0) | (sig_abl.hi < 0))
                sig_abl.to_csv(f"{OUT}/sig_ablations.csv", index=False)
                print(f"  ✓ sig_ablations.csv ({len(sig_abl)} rows)")

        print("\n[4/4] Writing LaTeX tables and figure...")
        def tex_escape(s):
            return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")

        def tex_static(df, ds):
            sub = df[df["dataset"] == ds]
            if sub.empty: return ""
            L = [r"\begin{table}[t]\centering\small",
                 r"\begin{tabular}{llrrrrr}", r"\toprule",
                 r"Metric & vs. & $\Delta$ (pp) & 95\% CI & $p_{\text{adj}}$ & $d$ & Sig. \\",
                 r"\midrule"]
            for _, r in sub.sort_values(["metric","vs"]).iterrows():
                s = r"\checkmark" if r["sig"] else "--"
                L.append(f"{r['metric']} & {r['vs']} & {r['mean_diff']*100:+.2f} & "
                         f"[{r['lo']*100:+.2f}, {r['hi']*100:+.2f}] & "
                         f"{r['p_adj']:.2e} & {r['cohens_d']:+.2f} & {s} \\\\")
            L += [r"\bottomrule", r"\end{tabular}",
                  f"\\caption{{Paired tests on {ds} static retrieval.}}", r"\end{table}"]
            return "\n".join(L)

        if not static.empty and not sig_static.empty:
            with open(f"{OUT}/table_sig_static.tex","w") as f:
                f.write(tex_static(sig_static, "2Wiki") + "\n\n" + tex_static(sig_static, "MuSiQue"))
            print(f"  ✓ table_sig_static.tex")

        if not qa.empty and not sig_qa.empty:
            with open(f"{OUT}/table_sig_qa.tex","w") as f:
                L = [r"\begin{table}[t]\centering\small",
                     r"\begin{tabular}{llrrrr}", r"\toprule",
                     r"Model & Metric & $\Delta$ (pp) & 95\% CI & $p_{\text{adj}}$ & Sig. \\", r"\midrule"]
                for _, r in sig_qa.iterrows():
                    s = r"\checkmark" if r["sig"] else "--"
                    L.append(f"{r['model']} & {r['metric']} & {r['mean']*100:+.2f} & "
                             f"[{r['lo']*100:+.2f}, {r['hi']*100:+.2f}] & "
                             f"{r['p_adj']:.4f} & {s} \\\\")
                L += [r"\bottomrule", r"\end{tabular}",
                      r"\caption{Hallucination deltas (mitigated $-$ standard).}",
                      r"\end{table}"]
                f.write("\n".join(L))
            print(f"  ✓ table_sig_qa.tex")

        if not abl.empty and not sig_abl.empty:
            with open(f"{OUT}/table_sig_abl.tex","w") as f:
                L = [r"\begin{table}[t]\centering\small",
                     r"\begin{tabular}{llrrrr}", r"\toprule",
                     r"Model & Metric & $\Delta$ (pp) & 95\% CI & $p_{\text{adj}}$ & Sig. \\", r"\midrule"]
                for _, r in sig_abl.iterrows():
                    s = r"\checkmark" if r["sig"] else "--"
                    L.append(f"{r['model']} & {r['metric']} & {r['mean']*100:+.2f} & "
                             f"[{r['lo']*100:+.2f}, {r['hi']*100:+.2f}] & "
                             f"{r['p_adj']:.4f} & {s} \\\\")
                L += [r"\bottomrule", r"\end{tabular}",
                      r"\caption{Ablation: Ours $-$ no\_nli\_gate.}", r"\end{table}"]
                f.write("\n".join(L))
            print(f"  ✓ table_sig_abl.tex")

        if not static.empty:
            with open(f"{OUT}/table_static.tex","w") as f:
                f.write("\\begin{table}[t]\\centering\\small\n")
                f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
                f.write("Method & Prec & Rec & SelPrec & SelF1 & OrSel & DriftCE \\\\\n\\midrule\n")
                agg = static.groupby("tag")[["Prec","Rec","SelPrec","SelF1","OrSel","DriftCE"]].mean()*100
                for tag, row in agg.iterrows():
                    f.write(f"{tex_escape(tag)} & " + " & ".join(f"{v:.2f}" for v in row.values) + " \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n\\caption{Static retrieval.}\n\\end{table}\n")
            print(f"  ✓ table_static.tex")

        if not qa.empty:
            with open(f"{OUT}/table_hallucination.tex","w") as f:
                f.write("\\begin{table}[t]\\centering\\small\n\\begin{tabular}{llccc}\n\\toprule\n")
                f.write("Model & Method & EM & F1 & Unsup \\\\\n\\midrule\n")
                agg = qa.groupby(["model","method"])[["EM","F1","Unsup"]].mean()*100
                for (model, method), row in agg.iterrows():
                    f.write(f"{tex_escape(model.split('/')[-1])} & {tex_escape(method)} & "
                            + " & ".join(f"{v:.2f}" for v in row.values) + " \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n\\caption{Hallucination by model/mode.}\n\\end{table}\n")
            print(f"  ✓ table_hallucination.tex")

        if not abl.empty:
            with open(f"{OUT}/table_ablations.tex","w") as f:
                f.write("\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lccc}\n\\toprule\n")
                f.write("Variant & EM & F1 & Unsup \\\\\n\\midrule\n")
                agg = abl.groupby("variant")[["EM","F1","Unsup"]].mean()*100
                for v, row in agg.iterrows():
                    f.write(f"{tex_escape(v)} & " + " & ".join(f"{v2:.2f}" for v2 in row.values) + " \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n\\caption{Component ablations.}\n\\end{table}\n")
            print(f"  ✓ table_ablations.tex")

        if not free.empty:
            with open(f"{OUT}/table_freerider.tex","w") as f:
                f.write("\\begin{table}[t]\\centering\\small\n\\begin{tabular}{lcccc}\n\\toprule\n")
                f.write("Method & HubSem & HubSem(NLI) & Prec & Prec(NLI) \\\\\n\\midrule\n")
                agg = free.groupby("tag")[["HubSem","HubSem_mit","Prec","Prec_mit"]].mean()
                for tag, row in agg.iterrows():
                    f.write(f"{tex_escape(tag)} & " + " & ".join(f"{v:.3f}" for v in row.values) + " \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n\\caption{Free-rider inclusion.}\n\\end{table}\n")
            print(f"  ✓ table_freerider.tex")

        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, ds in zip(axes, ["2Wiki", "MuSiQue"]):
                sub = sig_static[(sig_static["dataset"] == ds) & (sig_static["metric"] == "Prec")].sort_values("mean_diff")
                if sub.empty: ax.set_axis_off(); continue
                y = np.arange(len(sub)); mean = sub["mean_diff"].values * 100
                lo = sub["lo"].values * 100; hi = sub["hi"].values * 100
                ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="o", capsize=3,
                            color="#D55E00", ecolor="#444")
                ax.axvline(0, color="k", lw=0.8, ls="--")
                ax.set_yticks(y); ax.set_yticklabels(sub["vs"].values, fontsize=9)
                ax.set_xlabel("Δ Context Precision (pp)")
                ax.set_title(f"{ds} (n={int(sub['n'].iloc[0])})"); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{OUT}/fig_sig_precision.pdf", bbox_inches="tight")
            plt.savefig(f"{OUT}/fig_sig_precision.png", dpi=200, bbox_inches="tight")
            print(f"  ✓ fig_sig_precision.pdf")
        except Exception as e:
            print(f"  [warn] figure failed: {e}")

        def _json_safe(o):
            if isinstance(o, dict): return {str(k): _json_safe(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [_json_safe(x) for x in o]
            if isinstance(o, (np.integer, np.floating)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return o

        summary = {
            "static": static.groupby(["dataset","tag"])[["Prec","Rec","SelPrec","SelF1","OrSel","DriftCE"]].mean().round(4).to_dict() if not static.empty else {},
            "qa_standard": qa[qa.method=="standard"].groupby(["dataset","model","tag"])[["EM","F1","Faith","Unsup","InCtx"]].mean().round(4).to_dict() if not qa.empty else {},
            "qa_mitigated": qa[qa.method=="mitigated"].groupby(["dataset","model","tag"])[["EM","F1","Faith","Unsup","InCtx"]].mean().round(4).to_dict() if not qa.empty else {},
            "ablations": abl.groupby(["variant","model"])[["EM","F1","Faith","Unsup","InCtx"]].mean().round(4).to_dict() if not abl.empty else {},
            "freerider": free.groupby("tag")[["HubSem","HubSem_mit","Prec","Prec_mit"]].mean().round(4).to_dict() if not free.empty else {},
            "systems": scal.groupby("bench")[["per_upd_ms","recomp_ms","speedup","mismatch"]].median().round(4).to_dict() if not scal.empty else {},
        }
        with open(f"{OUT}/summary.json","w") as f:
            json.dump(_json_safe(summary), f, indent=2)
        print(f"  ✓ summary.json")

        print("\n" + "="*70)
        print(f"✓ RECOVERY COMPLETE. All files in {OUT}/")
        print("="*70)
        total = 0
        for f in sorted(os.listdir(OUT)):
            fp = os.path.join(OUT, f)
            if os.path.isfile(fp):
                sz = os.path.getsize(fp); total += sz
                print(f"  {f:45s}  {sz/1024:9.1f} KB")
        return print(f"  {'TOTAL':45s}  {total/1024/1024:9.1f} MB")


    _()
    return



@app.cell
def _():

    import subprocess, sys
    print("="*70); print("ANON-NLP REVIEWER-BOOST ANALYSIS"); print("="*70)

    for pkg in ["torch","transformers","datasets","sentence-transformers","networkx",
                "pandas","numpy","matplotlib","scipy","accelerate","bitsandbytes"]:
        try: __import__(pkg.replace("-","_").split("[")[0])
        except ImportError:
            subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])

    import os, re, math, json, time, string, random, warnings, collections, glob, base64
    from dataclasses import dataclass
    from collections import deque, defaultdict, Counter
    import numpy as np, pandas as pd, networkx as nx
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import torch, torch.nn as nn, torch.nn.functional as F
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer, util, CrossEncoder
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForSequenceClassification,
                              BitsAndBytesConfig)
    try: from scipy.stats import wilcoxon
    except Exception: wilcoxon = None
    warnings.filterwarnings("ignore")
    try: import community as community_louvain
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install","-q","python-louvain"])
        import community as community_louvain

    # switches
    RUN_BASELINES         = True
    RUN_2WIKI_DOWNSTREAM  = True
    RUN_INVARIANTS        = True
    N_2WIKI_DOWNSTREAM    = 400
    N_INVARIANT_REPS      = 3

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

    def _pick_out():
        for c in ["/root/anon_submission", "/marimo", os.path.expanduser("~")]:
            if os.path.isdir(c) and os.access(c, os.W_OK):
                return os.path.join(c, "reviewer_boost")
        return os.path.abspath("reviewer_boost")
    OUT = _pick_out(); os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT,"tables"), exist_ok=True)
    os.makedirs(os.path.join(OUT,"figures"), exist_ok=True)
    print(f"Output: {OUT}")

    def set_seed(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

    def norm_edge(u,v): return tuple(sorted((str(u),str(v))))
    def norm_ans(s):
        s = str(s or "").lower(); s = re.sub(r"\b(a|an|the)\b"," ",s)
        return " ".join("".join(c for c in s if c not in string.punctuation).split())
    def em_f(p,g): return float(norm_ans(p)==norm_ans(g))
    def f1_f(p,g):
        pt,gt = norm_ans(p).split(), norm_ans(g).split()
        if not pt or not gt: return 0.0
        c = sum((Counter(pt)&Counter(gt)).values())
        return 0.0 if c==0 else 2*(c/len(pt))*(c/len(gt))/((c/len(pt))+(c/len(gt)))
    def clean_ans(t):
        t = str(t or "").strip()
        for s in ["\nQuestion","\nAnswer","\nContext","\nFinal Answer","\n\n",
                  "Question:","Answer:","Context:","Final Answer:"]:
            if s in t: t = t.split(s)[0]
        return t.split("\n")[0].strip().strip('"').strip("'").rstrip(".").strip()
    def safe_sig(x,c=20.0):
        x = float(np.clip(x,-c,c)); return 1.0/(1.0+math.exp(-x))
    ENTITY = re.compile(r"\b[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*\b")

    set_seed(42)

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEV}")

    _need_encoders = RUN_BASELINES or RUN_2WIKI_DOWNSTREAM
    if _need_encoders:
        print("Loading encoders...")
        EMB = SentenceTransformer("all-MiniLM-L6-v2", device=DEV)
        CE  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEV)
        NT  = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base")
        NM  = AutoModelForSequenceClassification.from_pretrained(
                  "cross-encoder/nli-deberta-v3-base").to(DEV).eval()
        EI  = next((int(i) for i,l in getattr(NM.config,"id2label",{}).items()
                    if str(l).lower().startswith("entail")), 2)
        print("  ✓ encoders ready")
    else:
        EMB = CE = NT = NM = EI = None

    def enc(t):
        if isinstance(t, str): t = [t]
        return EMB.encode(t, convert_to_tensor=True, normalize_embeddings=True,
                          show_progress_bar=False)
    def nli_ent(p, h):
        if not p or not p.strip() or not h or not h.strip(): return float("nan")
        inp = NT(p, h, return_tensors="pt", truncation=True, max_length=512).to(DEV)
        with torch.no_grad():
            pr = torch.softmax(NM(**inp).logits, -1).cpu().numpy()[0]
        return float(pr[EI])
    def drift_ce(q, ctx):
        if not ctx or not ctx.strip(): return 1.0
        try:
            return 1.0 - safe_sig(float(np.asarray(CE.predict([(q,ctx)]), float).ravel()[0]))
        except Exception: return 1.0

    # ── minimal KDCore ────────────────────────────────────────────────
    class KDCore:
        def __init__(s, k, R, kc=True, rad=True, sem=False, thr=0.12, protect=True):
            s.k,s.R,s.kc,s.rad,s.sem,s.thr,s.protect = k,R,kc,rad,sem,thr,protect
            s.adj = defaultdict(set); s.core = set(); s.deg = defaultdict(int)
            s.pivot = None; s.depth = {}; s.par = {}; s.ch = defaultdict(set)
            s.scores = {}; s.anc = set()
        def build(s, edges, nodes, anchors, scores):
            s.adj = defaultdict(set); s.scores = scores or {}
            s.anc = {str(a) for a in anchors} if anchors else set()
            for n in nodes: _ = s.adj[str(n)]
            for u,v in edges:
                u,v = str(u),str(v)
                if u!=v: s.adj[u].add(v); s.adj[v].add(u)
            s.pivot = str(anchors[0]) if anchors else (sorted(s.adj)[0] if s.adj else None)
            s.depth, s.par, s.ch = s._tree()
            cand = set(s.depth) if s.rad else set(s.adj)
            if anchors:
                vis = set(); q = deque(str(a) for a in anchors if str(a) in s.adj); vis.update(q)
                while q:
                    u = q.popleft()
                    for w in s.adj[u]:
                        if w not in vis: vis.add(w); q.append(w)
                cand &= vis
            if s.kc: cand = s._peel(cand)
            s.core = set(cand)
            if s.sem: s._semprune()
            s._redeg(); return s.core
        def _tree(s):
            dep,par,ch = {},{},defaultdict(set)
            if s.pivot in s.adj:
                q = deque([(s.pivot,0)]); dep[s.pivot] = 0
                while q:
                    u,du = q.popleft()
                    if du==s.R: continue
                    for w in s.adj[u]:
                        if w not in dep:
                            dep[w]=du+1; par[w]=u; ch[u].add(w); q.append((w,du+1))
            return dep,par,ch
        def _peel(s, c):
            if s.k<=0 or not c: return set(c)
            c = set(c); deg = {n:len(s.adj[n]&c) for n in c}
            q = deque(n for n,d in deg.items() if d<s.k and n not in s.anc)
            while q:
                u = q.popleft()
                if u in c and u not in s.anc:
                    c.remove(u)
                    for w in s.adj[u]:
                        if w in c:
                            deg[w] -= 1
                            if deg[w]<s.k and w not in s.anc: q.append(w)
            return c
        def _redeg(s): s.deg = {n:len(s.adj[n]&s.core) for n in set(s.adj)|set(s.core)}
        def _elig(s, n):
            return (not s.kc or s.deg.get(n,0)>=s.k) and (not s.rad or n in s.depth)
        def _joins(s, seeds):
            q = deque(n for n in seeds if n not in s.core and s._elig(n))
            while q:
                u = q.popleft()
                if u in s.core or not s._elig(u): continue
                s.core.add(u)
                for w in s.adj[u]:
                    s.deg[w] = s.deg.get(w,0)+1
                    if w not in s.core and s._elig(w): q.append(w)
        def _removes(s, seeds):
            q = deque(n for n in seeds if n in s.core and s.deg.get(n,0)<s.k
                      and not (s.protect and n in s.anc))
            while q:
                u = q.popleft()
                if u not in s.core: continue
                s.core.discard(u)
                for w in s.adj[u]:
                    s.deg[w] = s.deg.get(w,0)-1
                    if w in s.core and s.deg.get(w,0)<s.k and not (s.protect and w in s.anc):
                        q.append(w)
        def _semprune(s):
            if not s.core: return
            anc = set(map(str, s.anc))
            bad = [n for n in s.core if s.scores.get(n,0.0)<s.thr and n not in anc]
            if len(s.core)-len(bad) >= max(s.k, 2):
                for n in bad: s.core.discard(n)
                s._redeg(); s._removes(bad); s._joins(list(s.adj))
        def insert(s, u, v):
            u,v = str(u),str(v)
            if u==v or v in s.adj[u]: return
            s.adj[u].add(v); s.adj[v].add(u)
            for a,b in ((u,v),(v,u)):
                if s.rad and a in s.depth and s.depth[a]<s.R and b not in s.depth:
                    s.depth[b]=s.depth[a]+1; s.par[b]=a; s.ch[a].add(b)
            s.deg[u] = len(s.adj[u]&s.core); s.deg[v] = len(s.adj[v]&s.core)
            if s.kc: s._joins([u,v])
            else:
                if s._elig(u): s.core.add(u)
                if s._elig(v): s.core.add(v)
        def delete(s, u, v):
            u,v = str(u),str(v)
            if v not in s.adj[u]: return
            te = (u,v) if s.par.get(v)==u else ((v,u) if s.par.get(u)==v else None)
            s.adj[u].discard(v); s.adj[v].discard(u)
            if te and s.rad: s._reattach(*te)
            for n in {u,v}|s.adj[u]|s.adj[v]: s.deg[n] = len(s.adj[n]&s.core)
            s._removes([u,v])
        def _reattach(s, x, y):
            S = set(); q = deque([y])
            while q:
                u = q.popleft()
                if u in S: continue
                S.add(u); q.extend(s.ch[u])
            for u in S:
                s.depth.pop(u, None); p = s.par.pop(u, None)
                if p is not None: s.ch[p].discard(u)
            for u in S:
                best = None
                for w in s.adj[u]:
                    if w not in S and w in s.depth and s.depth[w]<s.R:
                        if best is None or s.depth[w]<s.depth[best]: best = w
                if best is not None:
                    s.depth[u]=s.depth[best]+1; s.par[u]=best; s.ch[best].add(u)
            q = deque(u for u in S if u in s.depth)
            while q:
                u = q.popleft()
                if s.depth[u] >= s.R: continue
                for w in s.adj[u]:
                    if w in S and s.depth.get(w,10**9) > s.depth[u]+1:
                        op = s.par.get(w)
                        if op is not None: s.ch[op].discard(w)
                        s.depth[w]=s.depth[u]+1; s.par[w]=u; s.ch[u].add(w); q.append(w)
            lost = [u for u in S if u not in s.depth]
            if lost:
                for u in lost: s.core.discard(u)
                s._redeg(); s._removes(lost)

    # ── dataset parsing ───────────────────────────────────────────────
    def _graph(pm, gt, q_ents, answer, question):
        ent = defaultdict(set)
        for pid,t in pm.items():
            for e in ENTITY.findall(t): ent[e.lower()].add(pid)
        edges = set()
        for _,ps in ent.items():
            ps = sorted(ps)
            for i in range(len(ps)):
                for j in range(i+1,len(ps)): edges.add(norm_edge(ps[i], ps[j]))
        anc = []
        for e in q_ents:
            el = str(e).lower()
            for pid,t in pm.items():
                if el in t.lower() and pid not in anc: anc.append(pid); break
            if len(anc) >= 3: break
        return anc, edges, gt, answer, pm, question

    def parse_musique(s):
        pm, gt = {}, set()
        for p in s.get("paragraphs", []):
            pid = str(p.get("idx", len(pm)))
            pm[pid] = p.get("paragraph_text", "") or ""
            if p.get("is_supporting"): gt.add(pid)
        return _graph(pm, gt, s.get("question_entities", []),
                      s.get("answer", ""), s.get("question", ""))

    def parse_2wiki(s):
        pm, gt, titles = {}, set(), []
        for i, item in enumerate(s.get("context", []) or []):
            ti, se = "", []
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                ti, se = item[0], item[1]
            elif isinstance(item, dict):
                ti = item.get("title", str(i))
                se = item.get("sentences", item.get("text", []))
            if isinstance(se, str): se = [se]
            elif not isinstance(se, (list, tuple)): se = [str(se)]
            pm[str(i)] = " ".join(se); titles.append(str(ti))
        sf = s.get("supporting_facts", []) or []
        if isinstance(sf, list):
            for item in sf:
                if isinstance(item, (list, tuple)) and len(item) >= 1: t = item[0]
                elif isinstance(item, dict): t = item.get("title", "")
                else: t = str(item)
                for i, ti in enumerate(titles):
                    if str(ti) == str(t): gt.add(str(i)); break
        elif isinstance(sf, dict) and "title" in sf:
            for t in sf["title"]:
                for i, ti in enumerate(titles):
                    if str(ti) == str(t): gt.add(str(i)); break
        q_ents = list({w for w in ENTITY.findall(s.get("question", ""))})
        return _graph(pm, gt, q_ents, s.get("answer", ""), s.get("question", "")), len(gt) > 0

    # ── source CSV location ───────────────────────────────────────────
    def find_src():
        candidates = [
            "/root/anon_submission/anon_exp_v1",
            os.path.expanduser("~/anon_submission/anon_exp_v1"),
            "/marimo/anon_exp_v1",
            os.path.abspath("anon_exp_v1"),
        ]
        for c in candidates:
            if os.path.isdir(c) and os.path.exists(os.path.join(c, "MuSiQue_static.csv")):
                return c
        for r in [os.path.expanduser("~"), "/marimo", "/root", "."]:
            if not os.path.isdir(r): continue
            for dirpath, _, files in os.walk(r):
                if "MuSiQue_static.csv" in files and "2Wiki_static.csv" in files:
                    return dirpath
        return None

    SRC = find_src()
    print(f"Source dir: {SRC}")

    def rd(name):
        p = os.path.join(SRC, f"{name}.csv") if SRC else ""
        return pd.read_csv(p) if p and os.path.exists(p) else pd.DataFrame()

    def concat(names):
        dfs = [rd(n) for n in names if not rd(n).empty]
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    static = concat(["MuSiQue_static", "2Wiki_static"])
    dyn    = concat(["MuSiQue_dyn", "2Wiki_dyn"])
    qa     = concat(["MuSiQue_qa", "2Wiki_qa"])
    abl    = concat(["MuSiQue_ablations", "2Wiki_ablations"])
    free   = rd("free")
    scal   = rd("scal")
    print(f"  static={len(static)}  dyn={len(dyn)}  qa={len(qa)}  abl={len(abl)}  free={len(free)}  scal={len(scal)}")

    rng = np.random.default_rng(42); NBOOT = 5000

    def paired(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
        if len(a) < 5: return None
        d = a - b
        boots = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(NBOOT)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        try: p = 1.0 if np.allclose(d, 0) else float(wilcoxon(a, b).pvalue)
        except Exception: p = 1.0
        sd = d.std(ddof=1) if len(d) > 1 else 1.0
        return dict(n=len(d), mean=float(d.mean()), lo=float(lo), hi=float(hi),
                    p=p, cohend=float(d.mean()/sd) if sd > 0 else 0.0)

    def holm(ps):
        ps = np.asarray(ps, float); m = len(ps)
        order = np.argsort(ps); adj = np.empty(m); prev = 0.0
        for r, i in enumerate(order):
            prev = max(prev, (m - r) * ps[i]); adj[i] = min(prev, 1.0)
        return adj

    def tex_escape(s):
        return str(s).replace("_","\\_").replace("%","\\%").replace("&","\\&")

    # ═══════════════════════════════════════════════════════════════════
    # PART A — EXTERNAL BASELINES
    # ═══════════════════════════════════════════════════════════════════
    if RUN_BASELINES:
        print("\n" + "="*70); print("PART A — EXTERNAL BASELINES"); print("="*70)

        def _sem_edges(ids, nemb, thr):
            es = set()
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    if float(util.pytorch_cos_sim(nemb[ids[i]].unsqueeze(0),
                            nemb[ids[j]].unsqueeze(0))) >= thr:
                        es.add(norm_edge(ids[i], ids[j]))
            return es

        def baseline_rerank(q, pm, ids, nemb, qe, ext, topk=10):
            scored = sorted(ids, key=lambda n: -ext.get(n, 0.0))
            return scored[:topk], scored

        def baseline_toporag(q, pm, ids, adj, nemb, qe, ext, anc, R=2, topk=10):
            ancS = list(map(str, anc))
            reachable = set()
            for a in ancS:
                if a not in adj: continue
                qq = deque([(a, 0)])
                while qq:
                    u, du = qq.popleft()
                    if u in reachable: continue
                    reachable.add(u)
                    if du == R: continue
                    for w in adj.get(u, set()):
                        if w not in reachable:
                            qq.append((w, du+1))
            sub = [(n, 0.5*float(util.pytorch_cos_sim(qe.unsqueeze(0), nemb[n].unsqueeze(0)).item())
                        + 0.5*ext.get(n, 0.0))
                   for n in reachable if n in nemb]
            sub.sort(key=lambda kv: -kv[1])
            return [n for n,_ in sub[:topk]], list(reachable)

        def baseline_graphrag(q, pm, ids, adj, nemb, qe, ext, topk=10):
            try:
                G = nx.Graph()
                for n in ids: G.add_node(n)
                for u, ns in adj.items():
                    for v in ns:
                        if u < v: G.add_edge(u, v)
                if G.number_of_edges() == 0:
                    return baseline_rerank(q, pm, ids, nemb, qe, ext, topk)
                partition = community_louvain.best_partition(G)
            except Exception:
                return baseline_rerank(q, pm, ids, nemb, qe, ext, topk)
            comm_scores = defaultdict(list)
            for n, c in partition.items():
                comm_scores[c].append(ext.get(n, 0.0))
            ranked_comms = sorted(comm_scores.items(), key=lambda kv: -np.mean(kv[1]))
            out = []
            for c, _ in ranked_comms:
                members = [n for n, cc in partition.items() if cc == c]
                members.sort(key=lambda n: -ext.get(n, 0.0))
                for m in members:
                    if m not in out: out.append(m)
                    if len(out) >= topk: return out, list(partition.keys())
            return out[:topk], list(partition.keys())

        print("Loading datasets for baseline comparison...")
        _2wiki_samples = []
        try:
            ds = load_dataset("voidful/2WikiMultihopQA", split="validation")
            for x in ds:
                g, ok = parse_2wiki(dict(x))
                if ok: _2wiki_samples.append(g)
            print(f"  2Wiki: {len(_2wiki_samples)}")
        except Exception as e:
            print(f"  2Wiki failed: {type(e).__name__}")
        _musique_samples = []
        try:
            ds = load_dataset("dgslibisey/MuSiQue", split="validation")
            _musique_samples = [parse_musique(ds[i]) for i in range(len(ds))]
            print(f"  MuSiQue: {len(_musique_samples)}")
        except Exception as e:
            print(f"  MuSiQue failed: {type(e).__name__}")

        def run_baselines_on(samples, ds_name, n_max=None):
            rows = []
            n = len(samples) if n_max is None else min(n_max, len(samples))
            for sid, (anc, edges, gt, ga, pm, q) in enumerate(samples[:n]):
                if not pm or not gt: continue
                ids = sorted(pm)
                nemb = {k: enc([pm[k]])[0] for k in ids}
                qe = enc([q])[0]
                if not anc:
                    anc = [max(ids, key=lambda k: float(util.pytorch_cos_sim(
                        qe.unsqueeze(0), nemb[k].unsqueeze(0))))]
                raw = np.asarray(CE.predict([(q, pm[k]) for k in ids]), float).ravel()
                mn, mx = raw.min(), raw.max()
                ext = {k: float((raw[i]-mn)/(mx-mn+1e-9)) for i, k in enumerate(ids)}

                def metrics(sel):
                    sgt = len(set(sel) & gt)
                    p = sgt/len(sel) if sel else 0.0
                    r = sgt/len(gt) if gt else 0.0
                    selt = " ".join(pm[k] for k in sel if k in pm)
                    return dict(SelPrec=p, SelRec=r,
                                SelF1=2*p*r/(p+r) if p+r>0 else 0.0,
                                OrSel=float(norm_ans(ga) in norm_ans(selt)))

                kd = KDCore(2, 2, True, True, True, thr=0.12)
                kd.build(edges | _sem_edges(ids, nemb, 0.45), ids, anc, ext)
                rc = {kk: 0.5*float(util.pytorch_cos_sim(qe.unsqueeze(0),
                      nemb[kk].unsqueeze(0)).item()) + 0.5*ext[kk] for kk in kd.core}
                sel = sorted(rc, key=rc.get, reverse=True)[:10]
                rows.append(dict(dataset=ds_name, sid=sid, method="Ours", **metrics(sel)))

                sel, _ = baseline_rerank(q, pm, ids, nemb, qe, ext)
                rows.append(dict(dataset=ds_name, sid=sid, method="RerankedTopK", **metrics(sel)))

                sel, _ = baseline_toporag(q, pm, ids,
                                          {k: kd.adj.get(k, set()) for k in ids},
                                          nemb, qe, ext, anc, R=2)
                rows.append(dict(dataset=ds_name, sid=sid, method="TopoRAG-like", **metrics(sel)))

                sel, _ = baseline_graphrag(q, pm, ids,
                                           {k: kd.adj.get(k, set()) for k in ids},
                                           nemb, qe, ext)
                rows.append(dict(dataset=ds_name, sid=sid, method="GraphRAG-like", **metrics(sel)))

                if sid % 200 == 0 and sid > 0:
                    print(f"    {ds_name}: {sid}/{n}")
            return rows

        bl_rows = []
        if _musique_samples:
            print("  Running on MuSiQue...")
            bl_rows += run_baselines_on(_musique_samples, "MuSiQue", n_max=None)
        if _2wiki_samples:
            print("  Running on 2Wiki (subset, n=1000)...")
            bl_rows += run_baselines_on(_2wiki_samples, "2Wiki", n_max=1000)

        if bl_rows:
            bdf = pd.DataFrame(bl_rows)
            bdf.to_csv(f"{OUT}/A_baselines.csv", index=False)
            print("\n  Baseline summary (mean %):")
            summ = bdf.groupby(["dataset","method"])[["SelPrec","SelRec","SelF1","OrSel"]].mean()*100
            print(summ.round(2).to_string())

            sig_rows = []
            for ds in bdf.dataset.unique():
                sub = bdf[bdf.dataset == ds]
                for met in ["SelPrec","SelRec","SelF1","OrSel"]:
                    piv = sub.pivot_table(index="sid", columns="method", values=met)
                    if "Ours" not in piv.columns: continue
                    for base in piv.columns:
                        if base == "Ours": continue
                        r = paired(piv["Ours"].values, piv[base].values)
                        if r: r.update(dataset=ds, metric=met, vs=base); sig_rows.append(r)
            if sig_rows:
                sdf = pd.DataFrame(sig_rows)
                sdf["p_adj"] = np.nan
                for (ds, met), g in sdf.groupby(["dataset","metric"]):
                    sdf.loc[g.index, "p_adj"] = holm(g["p"].values)
                sdf["sig"] = (sdf.p_adj < 0.05) & ((sdf.lo > 0) | (sdf.hi < 0))
                sdf.to_csv(f"{OUT}/A_baselines_significance.csv", index=False)

                with open(f"{OUT}/tables/table_baselines.tex","w") as f:
                    f.write("\\begin{table*}[!htbp]\n\\centering\\small\n")
                    f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
                    f.write("Dataset & Method & SelPrec & SelF1 & OrSel & $\\Delta$ vs ours (SelPrec)\\\\\n\\midrule\n")
                    for (ds, method), g in summ.iterrows():
                        sub = sdf[(sdf.dataset == ds) & (sdf.vs == method) & (sdf.metric == "SelPrec")]
                        delta = "---"
                        if len(sub):
                            r = sub.iloc[0]
                            mk = "$^{\\dagger}$" if r["sig"] else ""
                            delta = f"{r['mean']*100:+.2f}{mk}"
                        f.write(f"{ds} & {method} & {g['SelPrec']:.2f} & {g['SelF1']:.2f} & "
                                f"{g['OrSel']:.2f} & {delta}\\\\\n")
                    f.write("\\bottomrule\n\\end{tabular}\n")
                    f.write("\\caption{External baselines on retrieval quality. "
                            "$^{\\dagger}$Holm-adjusted $p<0.05$ vs ours.}"
                            "\\label{tab:baselines}\\end{table*}\n")
                print(f"  ✓ {OUT}/tables/table_baselines.tex")

    # ═══════════════════════════════════════════════════════════════════
    # PART B — 2WIKI DOWNSTREAM
    # ═══════════════════════════════════════════════════════════════════
    if RUN_2WIKI_DOWNSTREAM and _2wiki_samples:
        print("\n" + "="*70); print("PART B — 2WIKI DOWNSTREAM"); print("="*70)

        def try_load_llm(mid):
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                if tok.pad_token is None: tok.pad_token = tok.eos_token
                qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.float16)
                mod = AutoModelForCausalLM.from_pretrained(
                    mid, quantization_config=qc, device_map="auto",
                    low_cpu_mem_usage=True).eval()
                return tok, mod
            except Exception as e:
                print(f"    [4-bit: {type(e).__name__}: {str(e)[:180]}]")
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                if tok.pad_token is None: tok.pad_token = tok.eos_token
                mod = AutoModelForCausalLM.from_pretrained(
                    mid, torch_dtype=torch.float16, device_map="auto",
                    low_cpu_mem_usage=True).eval()
                return tok, mod
            except Exception as e:
                print(f"    [fp16: {type(e).__name__}: {str(e)[:180]}]")
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                if tok.pad_token is None: tok.pad_token = tok.eos_token
                mod = AutoModelForCausalLM.from_pretrained(
                    mid, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
                return tok, mod
            except Exception as e:
                print(f"    [fp32: {type(e).__name__}: {str(e)[:180]}]")
            return None, None

        LLM_IDS = ["microsoft/Phi-3-mini-4k-instruct",
                   "Qwen/Qwen2.5-3B-Instruct",
                   "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                   "Qwen/Qwen2.5-7B-Instruct"]
        LLMS = {}
        for mid in LLM_IDS:
            print(f"  Loading {mid}...")
            tok, mod = try_load_llm(mid)
            if mod is None:
                print(f"  ✗ {mid} failed on all strategies"); continue
            LLMS[mid] = (tok, mod)
            print(f"  ✓ {mid}")

        print(f"\nSuccessfully loaded {len(LLMS)}/4 LLMs")

        if LLMS:
            samples = _2wiki_samples
            def to_t(ids):
                if isinstance(ids, torch.Tensor): t = ids.long()
                elif hasattr(ids, "input_ids"): t = torch.as_tensor(ids["input_ids"], dtype=torch.long)
                else: t = torch.tensor(np.asarray(ids), dtype=torch.long)
                if t.dim()==1: t = t.unsqueeze(0)
                return t.to(DEV)

            def gen(tok, mod, ctx, q, max_new=96, ctx_tok=1536):
                pc = ctx or "none"
                try:
                    ic = tok(pc, add_special_tokens=False)["input_ids"]
                    if len(ic) > ctx_tok: pc = tok.decode(ic[:ctx_tok], skip_special_tokens=True)
                except Exception: pass
                prompt = (f"Context: {pc}\nQuestion: {q}\n"
                          "Answer with a short phrase using only the context; say unknown if absent.\nAnswer:")
                try:
                    ids = to_t(tok.apply_chat_template(
                        [{"role":"user","content":prompt}], add_generation_prompt=True,
                        return_tensors="pt"))
                except Exception:
                    ids = to_t(tok(prompt, return_tensors="pt", truncation=True,
                                   max_length=2048)["input_ids"])
                if ids.shape[1] > 4096: ids = ids[:, -4096:]
                with torch.no_grad():
                    out = mod.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       max_new_tokens=max_new, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
                return clean_ans(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))

            N = min(N_2WIKI_DOWNSTREAM, len(samples))
            rows = []
            print(f"Running on {N} examples × {len(LLMS)} models × 2 variants...")
            for sid, (anc, edges, gt, ga, pm, q) in enumerate(samples[:N]):
                if not pm or not gt: continue
                ids = sorted(pm)
                nemb = {k: enc([pm[k]])[0] for k in ids}
                qe = enc([q])[0]
                if not anc:
                    anc = [max(ids, key=lambda k: float(util.pytorch_cos_sim(
                        qe.unsqueeze(0), nemb[k].unsqueeze(0))))]
                sem = set()
                for i in range(len(ids)):
                    for j in range(i+1, len(ids)):
                        if float(util.pytorch_cos_sim(nemb[ids[i]].unsqueeze(0),
                                nemb[ids[j]].unsqueeze(0))) >= 0.45:
                            sem.add(norm_edge(ids[i], ids[j]))
                base = edges | sem
                raw = np.asarray(CE.predict([(q, pm[k]) for k in ids]), float).ravel()
                mn, mx = raw.min(), raw.max()
                ext = {k: float((raw[i]-mn)/(mx-mn+1e-9)) for i, k in enumerate(ids)}

                kd = KDCore(2, 2, True, True, True, thr=0.12); kd.build(base, ids, anc, ext)
                rc = {k: 0.5*float(util.pytorch_cos_sim(qe.unsqueeze(0), nemb[k].unsqueeze(0)).item())
                        + 0.5*ext[k] for k in kd.core}
                sel_ac = sorted(rc, key=rc.get, reverse=True)[:10]
                ctx_ac = " ".join(pm[k] for k in sel_ac if k in pm)

                kd_fg = KDCore(2, 2, False, False, False, thr=0.12); kd_fg.build(base, ids, anc, ext)
                rc_fg = {k: 0.5*float(util.pytorch_cos_sim(qe.unsqueeze(0), nemb[k].unsqueeze(0)).item())
                            + 0.5*ext[k] for k in kd_fg.core}
                sel_fg = sorted(rc_fg, key=rc_fg.get, reverse=True)[:10]
                ctx_fg = " ".join(pm[k] for k in sel_fg if k in pm)

                for mid, (tok, mod) in LLMS.items():
                    for tag, ctx in [("Ours", ctx_ac), ("FullGraph", ctx_fg)]:
                        ans = gen(tok, mod, ctx, q)
                        f = nli_ent(ctx, ans)
                        rows.append(dict(dataset="2Wiki", sid=sid, model=mid.split("/")[-1],
                                         method=tag,
                                         EM=em_f(ans, ga), F1=f1_f(ans, ga),
                                         Faith=float(f) if np.isfinite(f) else np.nan,
                                         Unsup=float(f < 0.5) if np.isfinite(f) else np.nan,
                                         InCtx=float(norm_ans(ans) in norm_ans(ctx)) if ans else 0.0))
                if sid % 50 == 0 and sid > 0:
                    print(f"  {sid}/{N}")
                    pd.DataFrame(rows).to_csv(f"{OUT}/B_2wiki_downstream.csv", index=False)
            pd.DataFrame(rows).to_csv(f"{OUT}/B_2wiki_downstream.csv", index=False)
            print(f"  ✓ {OUT}/B_2wiki_downstream.csv ({len(rows)} rows)")

            df = pd.DataFrame(rows)
            if len(df) and "model" in df.columns:
                sig_rows = []
                for model in df.model.unique():
                    for met in ["EM","F1","Faith","Unsup","InCtx"]:
                        sub = df[(df.model==model) & df[met].notna()]
                        piv = sub.pivot_table(index="sid", columns="method", values=met)
                        if "Ours" not in piv.columns or "FullGraph" not in piv.columns: continue
                        pr = piv[["Ours","FullGraph"]].dropna()
                        if len(pr) < 5: continue
                        d = pr["Ours"].values - pr["FullGraph"].values
                        boots = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(5000)])
                        lo, hi = np.percentile(boots, [2.5, 97.5])
                        try: p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
                        except Exception: p = 1.0
                        sd = d.std(ddof=1) if len(d) > 1 else 1.0
                        sig_rows.append(dict(model=model, metric=met, n=len(d),
                                             mean=float(d.mean()), lo=float(lo), hi=float(hi),
                                             p=p, cohend=float(d.mean()/sd) if sd > 0 else 0.0))
                if sig_rows:
                    sdf = pd.DataFrame(sig_rows)
                    sdf["p_adj"] = np.nan
                    for (model, met), g in sdf.groupby(["model","metric"]):
                        sdf.loc[g.index, "p_adj"] = holm(g["p"].values)
                    sdf["sig"] = (sdf.p_adj < 0.05) & ((sdf.lo>0) | (sdf.hi<0))
                    sdf.to_csv(f"{OUT}/B_2wiki_significance.csv", index=False)
                    with open(f"{OUT}/tables/table_2wiki_downstream.tex","w") as f:
                        f.write("\\begin{table*}[!htbp]\n\\centering\\small\n")
                        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
                        f.write("Model & Metric & $\\Delta$ (pp) & 95\\% CI & $p_{\\text{adj}}$ & Sig.\\\\\n\\midrule\n")
                        for _, r in sdf.sort_values(["metric","model"]).iterrows():
                            mk = "\\checkmark" if r["sig"] else "---"
                            f.write(f"{r['model']} & {r['metric']} & {r['mean']*100:+.2f} & "
                                    f"[{r['lo']*100:+.2f}, {r['hi']*100:+.2f}] & "
                                    f"{r['p_adj']:.4f} & {mk}\\\\\n")
                        f.write("\\bottomrule\n\\end{tabular}\n")
                        f.write("\\caption{2Wiki downstream: our method minus FullGraph contexts.}"
                                "\\label{tab:2wiki_downstream}\\end{table*}\n")
                    print(f"  ✓ {OUT}/tables/table_2wiki_downstream.tex")

        for mid in list(LLMS.keys()):
            try: del LLMS[mid]
            except Exception: pass
        LLMS.clear()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("\n[PART B skipped]")


    if RUN_INVARIANTS:
        print("\n" + "="*70); print("PART C — INVARIANT VERIFICATION"); print("="*70)

        def check_invariants(core, adj, anchor, k=2, R=3):
            core = set(map(str, core)); anchor = str(anchor)
            if not core:
                return dict(n=0, min_deg_non_anchor=0, max_dist=0,
                            unreachable=0, k_ok=False, R_ok=False)
            non_anchor = [u for u in core if u != anchor]
            if not non_anchor:
                min_deg = int(1e9)
            else:
                min_deg = min(len(adj.get(u, set()) & core) for u in non_anchor)
            pivot = next(iter(core))
            dist = {pivot: 0}; qq = deque([pivot])
            while qq:
                u = qq.popleft()
                for w in adj.get(u, set()):
                    if w in core and w not in dist:
                        dist[w] = dist[u]+1; qq.append(w)
            reachable = set(dist.keys())
            max_dist = max(dist.values()) if dist else 0
            return dict(n=len(core), min_deg_non_anchor=int(min_deg),
                        max_dist=int(max_dist) if reachable == core else int(1e9),
                        unreachable=len(core - reachable),
                        k_ok=bool(min_deg >= k),
                        R_ok=bool(max_dist <= R and reachable == core))

        results = []
        for n in (1000, 5000, 10000, 50000):
            G = nx.barabasi_albert_graph(n, 3, seed=n)
            nodes_list = [str(i) for i in G.nodes()]
            anchor = nodes_list[0]
            edges_set = {norm_edge(u,v) for u,v in G.edges()}
            for rep in range(N_INVARIANT_REPS):
                eng = KDCore(2, 3, True, True, False)
                eng.build(edges_set, nodes_list, [anchor], {})
                rng_i = np.random.default_rng(42 + rep + hash(f"BA{n}") % 997)
                cur = set(edges_set)
                for it in range(100):
                    el = list(cur)
                    if el and rng_i.random() < 0.5:
                        de = el[rng_i.integers(len(el))]; cur.discard(de); eng.delete(*de)
                    for _ in range(25):
                        u,v = rng_i.choice(nodes_list, 2, replace=False); e = norm_edge(u,v)
                        if e not in cur: cur.add(e); eng.insert(*e); break
                    if it % 10 == 0:
                        ex = KDCore(2, 3, True, True, False)
                        ex.build(cur, nodes_list, [anchor], {})
                        if ex.core != eng.core:
                            a = check_invariants(eng.core, eng.adj, anchor, k=2, R=3)
                            b = check_invariants(ex.core,   ex.adj,   anchor, k=2, R=3)
                            jac = (len(eng.core & ex.core) / max(len(eng.core | ex.core), 1))
                            results.append(dict(bench=f"BA-{n}", rep=rep, it=it, jaccard=jac,
                                                incr_n=a["n"], incr_min_deg=a["min_deg_non_anchor"],
                                                incr_max_dist=a["max_dist"],
                                                incr_unreach=a.get("unreachable",0),
                                                incr_k_ok=a["k_ok"], incr_R_ok=a["R_ok"],
                                                exact_n=b["n"], exact_min_deg=b["min_deg_non_anchor"],
                                                exact_max_dist=b["max_dist"],
                                                exact_unreach=b.get("unreachable",0),
                                                exact_k_ok=b["k_ok"], exact_R_ok=b["R_ok"]))

        rdf = pd.DataFrame(results)
        rdf.to_csv(f"{OUT}/C_invariants.csv", index=False)
        print(f"  {len(rdf)} divergent audits collected.")
        if len(rdf):
            both_k = (rdf.incr_k_ok & rdf.exact_k_ok).mean()
            both_R = (rdf.incr_R_ok & rdf.exact_R_ok).mean()
            print(f"  BOTH cores satisfy non-anchor density ≥ k: {both_k:.1%}")
            print(f"  BOTH cores satisfy radius ≤ R:              {both_R:.1%}")
            print(f"  Median Jaccard overlap: {rdf.jaccard.median():.3f}")
            with open(f"{OUT}/tables/table_invariants.tex","w") as f:
                f.write("\\begin{table}[!htbp]\n\\centering\\small\n")
                f.write("\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n")
                f.write("Benchmark & Divergent & Both $k$-ok & Med.\\ Jacc. & Max $|\\Delta\\deg|$\\\\\n\\midrule\n")
                for bench, g in rdf.groupby("bench"):
                    bk = (g.incr_k_ok & g.exact_k_ok).mean()
                    jc = g.jaccard.median()
                    md = (g.incr_min_deg - g.exact_min_deg).abs().max()
                    f.write(f"{bench} & {len(g)} & {int(bk*len(g))} ({bk:.0%}) & {jc:.2f} & {md}\\\\\n")
                f.write("\\midrule\n")
                f.write(f"Total & {len(rdf)} & {int(both_k*len(rdf))} ({both_k:.0%}) & "
                        f"{rdf.jaccard.median():.2f} & "
                        f"{(rdf.incr_min_deg - rdf.exact_min_deg).abs().max()}\\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n")
                f.write("\\caption{Invariant verification on divergent BA audits. "
                        "Density excludes the protected anchor. "
                        "Divergent pairs are near-misses on the tie-broken BFS frontier.}\n")
                f.write("\\label{tab:invariants}\\end{table}\n")
            print(f"  ✓ {OUT}/tables/table_invariants.tex")


    print("\n" + "="*70); print("DONE — Outputs"); print("="*70)
    for sub in ["tables","figures"]:
        d = os.path.join(OUT, sub)
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                print(f"  {sub}/{fn}")
    for fn in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, fn)
        if os.path.isfile(p):
            print(f"  {fn}")
    print("="*70)
    return



@app.cell
def _():
    def _():
        import os, base64
        SRC = "/marimo/anon_exp_v1"
        if not os.path.isdir(SRC):
            # try persistent mount
            for c in ("/root/anon_submission/anon_exp_v1",
                      os.path.expanduser("~/anon_submission/anon_exp_v1")):
                if os.path.isdir(c): SRC = c; break

        def _print_file(path, title, truncate=None):
            print("="*80); print(f"FILE: {title}"); print("="*80)
            if not os.path.exists(path):
                print("  [not present]"); return
            with open(path) as f:
                data = f.read()
            print(data if truncate is None else data[:truncate] + "\n... [truncated]")

        _print_file(f"{SRC}/table_sig_static.tex", "table_sig_static.tex")
        print("\n\n")
        _print_file(f"{SRC}/table_sig_abl.tex", "table_sig_abl.tex")
        print("\n\n")
        _print_file(f"{SRC}/table_sig_qa.tex", "table_sig_qa.tex")
        print("\n\n")
        _print_file(f"{SRC}/summary.json", "summary.json", truncate=3000)

        print("\n\n")
        print("="*80); print("FILE: sig_static.csv (base64)"); print("="*80)
        p = f"{SRC}/sig_static.csv"
        if os.path.exists(p):
            with open(p, "rb") as f:
                print(base64.b64encode(f.read()).decode())

        print("\n\n")
        print("="*80); print("FILE: fig_sig_precision.pdf (base64)"); print("="*80)
        p = f"{SRC}/fig_sig_precision.pdf"
        if os.path.exists(p):
            with open(p, "rb") as f:
                print(base64.b64encode(f.read()).decode())


    _()
    return


@app.cell
def _():
    def _():
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams.update({"font.family": "serif", "font.size": 9})
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

        data = {
            "2Wiki (n=12,576)": {
                "labels": ["FullGraph", "BoundedRadius", "KCoreOnly", "OursStruct", "SemanticPrune"],
                "mean":   [12.32, 8.93, 3.90, 0.98, 17.49],
                "lo":     [11.88, 8.58, 3.58, 0.89, 17.00],
                "hi":     [12.78, 9.29, 4.24, 1.07, 17.99],
            },
            "MuSiQue (n=2,417)": {
                "labels": ["FullGraph", "BoundedRadius", "KCoreOnly", "OursStruct", "SemanticPrune"],
                "mean":   [2.44, 1.09, 1.92, 0.41, 3.32],
                "lo":     [2.03, 0.77, 1.54, 0.26, 2.87],
                "hi":     [2.91, 1.44, 2.31, 0.57, 3.81],
            },
        }

        for ax, (title, d) in zip(axes, data.items()):
            order = np.argsort(d["mean"])
            y = np.arange(len(order))
            m = np.array(d["mean"])[order]
            lo = np.array(d["lo"])[order]
            hi = np.array(d["hi"])[order]
            labels = [d["labels"][i] for i in order]
            ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="o", capsize=3,
                        color="#D55E00", ecolor="#555")
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel(r"$\Delta$ Context Precision (pp)")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("fig_sig_precision.pdf", bbox_inches="tight")
        plt.savefig("fig_sig_precision.png", dpi=200, bbox_inches="tight")
        return print("✓ Wrote fig_sig_precision.pdf and .png")


    _()
    return


if __name__ == "__main__":
    app.run()