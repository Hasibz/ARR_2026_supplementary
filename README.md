# Supplementary Material — Anonymous ARR Submission
**Paper:** "When Better Retrieval Hurts: Dynamic k-Core Maintenance for
GraphRAG and the Retrieval-to-Generation Gap" (author identities withheld
for double-blind review)

**Contents:** code, figure sources, all paper tables, trained component
weights, and complete per-example outputs for every reported comparison.
Prompt templates and decoding configurations are transcribed in §7 and/or
embedded as constants in `code/`.

---

## 1. Bundle layout

supplementary/
├── README.md                  <- this file
├── code/                      <- pipeline + analysis scripts (anonymized)
├── figures/
│   ├── fig_sig_precision.pdf  <- source of main-paper Figure 3
│   └── fig01..fig10.png       <- sources of Appendix Figures 5-14
├── tables/                    <- LaTeX sources of Appendix Tables 3-12
│   └── table_{static,sig_static,sig_qa,sig_abl,ablations,
│                 freerider,hallucination}.tex
├── models/
│   ├── drift_rerank.pt        <- state_dict, drift reranker MLP
│   └── meta_ctrl.pt           <- state_dict, meta-controller MLP
└── outputs/
    ├── summary.json           <- machine-readable headline numbers
    ├── significance/
    │   ├── sig_static.csv     <- 60 paired tests (Table 1 / Fig. 3)
    │   ├── sig_qa.csv         <- hallucination deltas (mitigated-standard)
    │   ├── sig_ablations.csv  <- NLI-gate ablation (Table 9)
    │   └── B_2wiki_significance.csv <- paired tests for Table 2, 2Wiki column
    └── per_example/
        ├── 2Wiki_static.csv   <- 12,576 × 6 variants
        ├── MuSiQue_static.csv <- 2,417 × 6 variants
        ├── 2Wiki_dyn.csv      <- adversarial-update states (2Wiki)
        ├── MuSiQue_dyn.csv    <- adversarial-update states (MuSiQue)
        ├── MuSiQue_qa.csv     <- downstream, MuSiQue (all 6 variants)
        ├── MuSiQue_ablations.csv <- component ablations
        ├── B_2wiki_downstream.csv<- downstream, 2Wiki (Table 2, 2Wiki column)
        ├── C_invariants.csv   <- density-invariant audits (Table 8)
        │                         (original export name: C_invarants.csv)
        ├── free.csv           <- free-rider benchmark (n=300)
        └── scal.csv           <- BA scaling / systems timings

## 2. File → paper element map

| Artifact | Supports |
|---|---|
| `sig_static.csv` | Table 1, Figure 3, §5.1 paired tests |
| `2Wiki_static.csv`, `MuSiQue_static.csv` | per-example rows behind Tables 1, 4, 5 |
| `2Wiki_dyn.csv`, `MuSiQue_dyn.csv` | Table 6 (adversarial updates), §5.3 |
| `B_2wiki_downstream.csv` | **Table 2, 2Wiki column** (the paper's central negative result): per-example EM/F1/faithfulness/unsupported/in-context-answer for AuraCore vs FullGraph, 4 models × 2 prompting modes |
| `B_2wiki_significance.csv` | Table 2, 2Wiki column paired tests (the −4.5/−4.0/−6.75 pp EM and −5.4/−4.4/−7.9 pp F1 deltas) |
| `MuSiQue_qa.csv` | Table 2, MuSiQue column (filter variants to AuraCore/FullGraph); Table 12 |
| `sig_qa.csv` | §5.2 mitigation-stack ranges (−0.17…+0.83 pp) |
| `sig_ablations.csv` | Table 9 (NLI-gate ablation) |
| `MuSiQue_ablations.csv` | Table 10 (component ablations) |
| `C_invariants.csv` | Table 8 and Figure 12 (52 divergent BA audits) |
| `free.csv` | Table 7, Figure 9 (free-rider + guard stress test) |
| `scal.csv` | §5.3 scaling paragraph, Figure 11 |
| `summary.json` | all headline numbers (abstract, §5) |
| `tables/*.tex` | Appendix Tables 3–12 as typeset |
| `figures/*` | Figures 3 and 5–14 as typeset |
| `models/*.pt` | mitigation stack (§3.4, items ii–iii) |

## 3. Row-count and integrity checks

Expected data rows (header excluded). Counts marked *verify* should be
confirmed with `wc -l`; all others match the pipeline log.

| File | Expected rows | Derivation |
|---|---|---|
| `2Wiki_static.csv` | 75,456 | 12,576 × 6 variants |
| `MuSiQue_static.csv` | 14,502 | 2,417 × 6 variants |
| `2Wiki_dyn.csv` + `MuSiQue_dyn.csv` | 269,820 combined | pipeline SAMPLE COUNTS |
| `MuSiQue_qa.csv` | 19,200 | 400 × 4 models × 6 variants × 2 modes |
| `MuSiQue_ablations.csv` | 3,000 | 150 × 4 models × 5 variants |
| `B_2wiki_downstream.csv` | 6,400 *(verify)* | 400 × 4 models × 2 variants × 2 modes |
| `B_2wiki_significance.csv` | 40 *(verify)* | 4 models × 2 modes × 5 metrics |
| `C_invariants.csv` | 52 divergent audits *(verify; may include all 120 audits, of which 52 diverge)* | 3 seeds × 10 checkpoints × 4 sizes |
| `free.csv` | 1,800 | 300 × 6 variants |
| `scal.csv` | 15 | 5 benchmarks × 3 seeds |

    wc -l outputs/per_example/*.csv outputs/significance/*.csv

## 4. Reproducing the paper's numbers from this bundle

All code lives in a single self-configuring script; there are no external
config files. Hyperparameters (k=2, R=2, tau=0.12, semantic-edge threshold
0.45, top-10 nodes within 2,048 tokens), seeds (42, 7, 123), model IDs, and
the output directory are constants at the top of the script; edit them
there to reproduce alternative configurations.

Full pipeline (GPU, ~11-12 h on one RTX PRO 6000):

    python code/auracore_framework.py

Run end-to-end, it reproduces in order: the per-example CSVs
(outputs/per_example/), the paired significance tests
(outputs/significance/), the LaTeX table fragments (tables/),
figures/fig_sig_precision.pdf, and outputs/summary.json.

Statistics-only verification (CPU, minutes): the script's recovery stage
(steps [1/4]-[4/4] of its console output) reloads the bundled per-example
CSVs and recomputes sig_*.csv, tables/*.tex, the forest plot, and
summary.json without re-running retrieval or generation; see the script
header for its invocation.

## 5. Re-running the full pipeline (optional, GPU)

Models (HuggingFace IDs): `all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`,
`nli-deberta-v3-base`; generators `HuggingFaceTB/SmolLM2-1.7B-Instruct`,
`Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`,
`microsoft/Phi-3-mini-4k-instruct`. Hardware used: 1× NVIDIA RTX PRO 6000;
approx. 11–12 h wall time (dominated by 4-model generation). Datasets:
2WikiMultiHopQA and MuSiQue validation splits (2Wiki via the
`voidful/2WikiMultihopQA` mirror); HotpotQA excluded after a loader failure
(see paper Limitations).

    mv "code/auracore framework .py" code/auracore_framework.py

## 6. Trained components

`models/drift_rerank.pt` and `models/meta_ctrl.pt` are `state_dict`s of the
two small MLPs in mitigated mode (architectures in `code/modeling/mlp.py`),
trained by behavior cloning on the first 200 MuSiQue validation examples
(meta-controller; n=200) and 3,153 paragraph-level pairs (drift reranker;
371 positive) — the same split used for MuSiQue evaluation. This overlap is
disclosed in the paper Limitations and does not touch the 2Wiki negative
result. Load with:

    torch.load("models/drift_rerank.pt", map_location="cpu", weights_only=True)

## 7. Prompt templates and decoding configuration

- **NLI gate / faithfulness hypothesis template:**
  `"The answer to {question} is stated in this passage."`
  (premise = paragraph; threshold 0.20; top-3 fallback)
- **Generation template:** [PASTE VERBATIM from the code constant, e.g.
  `code/rag/prompts.py::QA_TEMPLATE`; if inline, copy exactly and cite
  file::symbol.]
- **Decoding:** greedy for all headline runs; self-consistency mode =
  3 samples at temperature 0.7, majority vote.
- **Context budget:** top-10 nodes, ≤2,048 tokens, ranked by
  0.5·sim + 0.5·rr.

## 8. Integrity notes (mirror the paper; read before reviewing numbers)

1. Density-invariant failures: 16/52 divergent BA audits, all at BA-10⁴
   (Table 8; per-audit rows in `C_invariants.csv`). Unresolved; flagged in
   the paper as its primary open problem.
2. Precision and recall co-vary by construction; no recall-matched control
   exists in this bundle (paper §4, Limitations).
3. Pooled mitigated-vs-standard tests are directional only (non-independent
   pooling); per-variant direction in Table 11.
4. Unsupported-claim rates of 85–98% reflect a conservative entailment
   criterion; interpret paired differences, not absolute levels.

## 9. Licenses & anonymity

Datasets and models are used under their original licenses. No author-identifying
information, local paths, or credentials are contained in this bundle.
