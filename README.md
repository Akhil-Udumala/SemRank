# 🛡️ SemRank: Unified Sandboxed Modular Search Ranking System

![Python](https://img.shields.io/badge/Python-3.11.4-blue)
![Runtime](https://img.shields.io/badge/Runtime-CPU--Only-green)
![Network](https://img.shields.io/badge/Sandbox-No%20Network-critical)
![Team](https://img.shields.io/badge/Team-Lucifer-purple)

**Track:** Redrob Hackathon — Semantic Ranker Challenge  
**Team:** Lucifer  
**Developer:** Udumala Akhil, Solo Developer  
**Platform:** MacBook Air, 8 cores, 16GB RAM, Python 3.11.4  
**Hosted Sandbox:** [Hugging Face Space](https://huggingface.co/spaces/UdumalaAkhil/redrob-ranker)

## Architectural Overview

SemRank is a hybrid offline/online candidate ranking system built for strict hackathon sandbox constraints. The design separates expensive semantic interpretation from the final reproducible ranking step.

```mermaid
graph TD
    %% Styling
    classDef phaseStyle fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef fileStyle fill:#e1f5fe,stroke:#0288d1,stroke-width:1px;
    classDef apiStyle fill:#fff3e0,stroke:#f57c00,stroke-width:1px;
    classDef coreStyle fill:#e8f5e9,stroke:#388e3c,stroke-width:1px;

    subgraph Phase1 [1. Offline Pre-Computation Phase - Network Enabled]
        A[data/candidates.jsonl] --> B[src/extract_features.py]
        B --> C[Gemini API Batch Processing]
        C --> D[artifacts/candidate_features.parquet]
    end
    class Phase1 phaseStyle; class A,D fileStyle; class C apiStyle;

    %% Boundary Line
    Boundary[- Network & Security Containment Boundary -]
    D --> Boundary
    Boundary --> E

    subgraph Phase2 [2. Online Sandbox Isolation Phase - Strictly Local / CPU Only]
        E[rank.py Entrypoint]
        F[data/candidates.jsonl] --> E
        
        E --> G[defenses.py<br/>Input Protection]
        E --> H[scoring.py<br/>Vectorized Matrix]
        E --> I[reasoning.py<br/>Justification Layer]
        
        G --> J[Deterministic Sorting & Ranking Loop]
        H --> J
        I --> J
        
        J --> K[team_id.csv Final Top 100]
    end
    class Phase2 phaseStyle; class F,K fileStyle; class E,G,H,I,J coreStyle;
    style Boundary fill:#ffebee,stroke:#c62828,stroke-width:2px,stroke-dasharray: 5 5;

The offline phase uses Gemini API batch processing to inspect candidate profile summaries, current titles, and career-history descriptions. It extracts compact semantic features such as production ML depth, ranking/evaluation experience, and factual reasoning snippets. These features are saved as a local Parquet artifact and are treated as static input during reproduction.

The online sandbox phase is fully deterministic and local. `rank.py` loads the candidate pool and the precomputed feature artifact, applies adversarial-profile defenses, computes a vectorized mathematical score matrix, performs mandatory deterministic sorting, and emits the final top-100 CSV. This phase makes no live LLM calls and contains no network-dependent ranking logic.

## Codebase Modularity

### `rank.py`

CLI entrypoint and orchestration layer. It parses `--candidates`, `--features`, and `--out`, loads local inputs, applies candidate filtering, calls the scoring matrix, sorts by `score` descending and `candidate_id` ascending, assigns ranks 1-100, and writes the submission CSV.

### `scoring.py`

Core engineering matrix. It evaluates candidate fit through weighted components covering career-domain relevance, title fit, production-shipping evidence, evaluation-framework experience, trusted skills, experience band, product-company signal, location fit, and Redrob behavioral signals. The implementation is vectorized with pandas/numpy for efficient processing over the 100K-candidate pool.

### `defenses.py`

Perimeter security layer. It rejects synthetic or structurally invalid profiles before scoring, including skill-duration impossibilities, current-role duration mismatches, and suspicious expert-skill anomalies. This layer prevents keyword-stuffed or impossible profiles from being boosted by semantic terms alone.

### `reasoning.py`

Natural-language justification layer for the top 100 rows. It first uses factual offline Gemini reasoning snippets when available, then falls back to deterministic local summaries assembled only from candidate record fields: title, years of experience, location, and listed technical skills. All reasoning strings are whitespace-normalized and capped to 500 characters.

### `src/extract_features.py`

Offline feature precomputation script. It streams candidates from `data/candidates.jsonl`, calls Gemini with a structured JSON schema, and writes `artifacts/candidate_features.parquet` for local sandbox consumption.

## Input Defenses & Trust Multipliers

SemRank treats resume text as untrusted input. Candidate summaries and career descriptions can contain noisy, adversarial, or keyword-stuffed content, so the ranking system avoids blindly trusting raw text frequency.

The defense layer validates structural consistency before scoring. A candidate is removed if any single skill duration exceeds the candidate's documented career duration, if career-history dates contradict declared role duration beyond tolerance, or if expert-level skill claims have impossible zero-duration patterns.

The scoring layer also applies trust multipliers. Low-fit titles receive localized dampening on career-domain and trusted-skill components, reducing the chance that non-engineering profiles with AI keyword stuffing outrank true production ML candidates. Skill evidence is weighted by proficiency, duration, and endorsements rather than raw keyword presence.

## Execution & Sandbox

Run the deterministic ranking step locally:

```bash
python rank.py --candidates ./data/candidates.jsonl --features ./artifacts/candidate_features.parquet --out ./team_id.csv
```

The ranking step is designed for CPU-only execution under the hackathon constraints. It does not call Gemini, hosted LLMs, embedding APIs, or external services. All semantic features must already exist in the local feature artifact before running `rank.py`.

The hosted Hugging Face Space provides a judge-facing verification surface for the same workflow. It demonstrates that the system can run in a controlled environment with local artifacts and deterministic output behavior, while preserving the no-network ranking contract expected during sandbox reproduction.

## Declarations

Gemini was used for offline batch semantic parsing and feature extraction only. Gemini is not part of the live ranking path.

Codex was used for inline coding, refactoring, implementation review, and documentation support. The final system architecture, deterministic scoring logic, honeypot defenses, and reproducible sandbox pipeline are implemented as project code in this repository.
