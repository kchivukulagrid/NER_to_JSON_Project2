
# NER JSON Studio: Fine-Tuned Qwen for Structured Entity Extraction

<p align="center">
  <img alt="Model" src="https://img.shields.io/badge/Model-Qwen2.5--1.5B-2563eb?style=for-the-badge">
  <img alt="Method" src="https://img.shields.io/badge/Tuning-LoRA-7c3aed?style=for-the-badge">
  <img alt="Best F1" src="https://img.shields.io/badge/Best%20F1-0.9066-059669?style=for-the-badge">
  <img alt="JSON Validity" src="https://img.shields.io/badge/JSON%20Validity-100%25-f59e0b?style=for-the-badge">
</p>

> Production-safe setup: `json_validate=yes`, `temperature=0.1`, constrained generation, with_defs adapter.


## 📌 Project Overview
This project builds a complete **NER to Structured JSON** pipeline using **Qwen2.5-1.5B + LoRA fine-tuning** on CoNLL-style data.

The system is designed to:
- extract entities from raw text,
- output clean schema-constrained JSON,
- compare decoding and data-prep strategies,
- support human correction + active learning with Gradio.

Target quality achieved:
- **F1 > 85%**
- **JSON validity = 100%** on key runs

---

## 🚀 Features
- **LoRA Fine-Tuning** of `Qwen/Qwen2.5-1.5B`
- **Structured JSON extraction** with schema normalization
- **Experiment sweeps**:
  - JSON validation yes/no
  - temperature (0.0 / 0.1 / 0.2)
  - output format (JSON / XML / plain)
  - generation mode (free / constrained)
  - data prep variants (with_defs / no_defs / syn_aug)
- **Gradio UI** for:
  - extraction,
  - correction,
  - active-learning export,
  - analytics dashboard
- **Active learning artifacts** persisted to JSONL
- **Plot dashboard** for reviewer presentation

---

## 🏗️ Installation & Setup

### Prerequisites
- Python `3.10-3.12` (recommended: `3.11`)
- macOS/Linux (MPS/CUDA/CPU supported via PyTorch setup)
- Git LFS (required to pull tracked model artifacts under `models/`)

### One-time Git LFS setup (per machine)
```bash
git lfs install
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Pull LFS-tracked model files (after clone)
```bash
git lfs pull
```

### Activate venv (recommended)
```bash
source venv/bin/activate
```

---

## ⚡ Run On Another Computer (Quick Start)

Use these exact commands on a fresh machine.

### 1. Clone and enter project
```bash
git clone https://github.com/kchivukulagrid/NER_to_JSON_Project2.git
cd NER_to_JSON_Project2
```

### 2. Create environment and install dependencies
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
git lfs install
git lfs pull
```

If `outlines_core` fails to build (usually on Python `3.13+`/`3.14`), use Python `3.11` for this project.  
Alternative (if you must keep Python `3.14`): install Rust first, then reinstall deps.

### 3. Run UI directly (no retraining required)
```bash
python -m src.gradio_correction_app
```

### 4. Rebuild and open analytics dashboard
```bash
python scripts/build_plotly_dashboard.py
open plots/index.html
```

### Notes for new users
- If adapter files are present in `experiments/with_defs_qwen2_5_1_5B`, UI runs with your tuned model.
- If adapter is missing, update `--adapter_path` when launching UI or keep your local adapter in the same path.
- UI runtime exports are written to:
  - `data/processed/active_learning/predictions_export.jsonl`
  - `data/processed/corrections/corrections.jsonl`
  - `data/processed/active_learning/cycle_records.jsonl`

---


## 📁 Project Structure
```text
NER_to_JSON_Project2/
├── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── task1_train.jsonl
│   │   ├── task1_val.jsonl
│   │   ├── task1_test.jsonl
│   │   ├── exports/
│   │   ├── corrections/
│   │   ├── active_learning/
│   │   ├── adversarial/
│   │   └── variants/
│   └── steering/
├── experiments/
│   ├── qwen2_5_1_5B_masked_tuned/
│   ├── data_prep_comparison/
│   ├── with_defs_qwen2_5_1_5B/
│   ├── no_defs_qwen2_5_1_5B/
│   ├── syn_aug_qwen2_5_1_5B/
│   ├── task1_constrained/
│   ├── task2_layer_importance/
│   ├── task3_steering/
│   ├── task4_adversarial/
│   └── task5_production/
├── models/
│   ├── gguf/
│   └── hf/
├── llama.cpp/
├── plots/
│   └── index.html
├── scripts/
│   ├── run_json_validity_f1_experiments.sh
│   ├── run_format_comparison_and_csv.sh
│   ├── run_generation_mode_comparison_and_csv.sh
│   ├── run_data_prep_comparison_and_csv.sh
│   ├── task1_run_benchmarks.sh
│   ├── build_plotly_dashboard.py
│   ├── build_data_prep_test_compare_csv.py
│   ├── build_review_queue.py
│   ├── export_corrections_jsonl.py
│   └── launch_correction_app.sh
├── src/
│   ├── core/
│   ├── tasks/
│   │   ├── task1_constrained/
│   │   ├── task2_layer_importance/
│   │   ├── task3_steering/
│   │   ├── task4_adversarial/
│   │   └── task5_production/
│   ├── train.py
│   ├── inference.py
│   ├── evaluation.py
│   └── gradio_correction_app.py
├── requirements.txt
└── README.md
```

---

## ✅ Project_2 Tasks

### 🔹 Task 1: Constrained Decoding for Structured NER
- Train LoRA for strict JSON schema output with offsets/confidence.
- Compare free vs constrained decoding quality and latency.

### 🔹 Task 2: Layer-wise Entity Type Importance
- Run logit-lens + per-layer ablation.
- Identify critical layers and evaluate selective LoRA placement.

### 🔹 Task 3: Boundary Steering (Gemma 2B)
- Extract activations on strict vs loose boundary sets (layers 12-16).
- Compute/apply steering vectors and evaluate boundary F1 tradeoffs.

### 🔹 Task 4: Adversarial Robustness
- Create adversarial categories (nested, abbrev, misspell, ambiguous, multilingual).
- Train with mixed original+adversarial data and compare pre/post robustness.

### 🔹 Task 5: Production Profiling & Quantization
- Profile inference memory.
- Benchmark Q4/Q5/Q8 quantizations for latency, memory, and F1.
- Measure concurrency (1/4/8/16) with p50/p95/p99 and throughput.

---

## 📊 Key Results Snapshot

### JSON validation × temperature (validation)
Source: `experiments/qwen2_5_1_5B_masked_tuned/json_validity_f1_experiment_results.csv`
- Peak validation F1 around: **0.9064**
- Production-safe stable config: **yes + 0.1** (0 repaired)

### Final test configuration comparison
Source: `experiments/qwen2_5_1_5B_masked_tuned/final_test_comparison.csv`
- `final_test_no_0p2`: F1 **0.8837** (higher, but repaired > 0)
- `final_test_yes_0p1`: F1 **0.8825** (cleaner, repaired = 0)

### Data prep impact on test
Source: `experiments/data_prep_comparison/data_prep_test_compare.csv`
- baseline test F1: **0.8837**
- with_defs test F1: **0.9066**

### Generation mode comparison (validation)
Source: `experiments/qwen2_5_1_5B_masked_tuned/gen_mode_comparison_temp_0p0_validate_yes_format_json.csv`
- constrained generation outperforms free generation

---

## 🏋️ Training & Inference

### Train
```bash
python -m src.train
```

### Inference (example)
```bash
python -m src.inference \
  --input_file data/processed/test.jsonl \
  --json_validate yes \
  --temperature 0.1 \
  --output_format json \
  --output_file data/processed/exports/final_test_yes_0p1.jsonl
```

### Evaluate
```bash
python -m src.evaluation \
  --input_file data/processed/exports/final_test_yes_0p1.jsonl \
  --output_file experiments/qwen2_5_1_5B_masked_tuned/final_test_yes_0p1_metrics.json
```

---

## 🧪 Experiment Commands

### 1. Parameters Experiment (JSON Validate + Temperature)
Purpose: compare `json_validate` (`yes/no`) across temperatures `0.0`, `0.1`, `0.2`.

```bash
bash scripts/run_json_validity_f1_experiments.sh
```

Primary output:
- `experiments/qwen2_5_1_5B_masked_tuned/json_validity_f1_experiment_results.csv`

### 2. Architecture Experiment: Output Format (JSON vs XML vs plain)
Purpose: compare output format behavior under one fixed decoding setup.
Recommended fixed setup: `--temperature 0.0 --json_validate yes`

```bash
bash scripts/run_format_comparison_and_csv.sh \
  --temperature 0.0 \
  --json_validate yes
```

Primary output:
- `experiments/qwen2_5_1_5B_masked_tuned/fmt_format_comparison_temp_0p0_validate_yes.csv`

### 3. Architecture Experiment: Generation Mode (Constrained vs Free)
Purpose: compare decoding strategy while keeping format/config fixed.
Recommended fixed setup: `--temperature 0.0 --json_validate yes --output_format json`

```bash
bash scripts/run_generation_mode_comparison_and_csv.sh \
  --temperature 0.0 \
  --json_validate yes \
  --output_format json
```

Primary output:
- `experiments/qwen2_5_1_5B_masked_tuned/gen_mode_comparison_temp_0p0_validate_yes_format_json.csv`

### 4. Data Prep Experiment (with_defs vs no_defs vs syn_aug)
Purpose: compare prompt/data variants after short retraining + evaluation.

```bash
bash scripts/run_data_prep_comparison_and_csv.sh \
  --epochs 2 \
  --generation_mode constrained \
  --temperature 0.0 \
  --json_validate yes
```

Primary outputs:
- `experiments/data_prep_comparison/data_prep_comparison_temp_0p0_mode_constrained.csv`
- `experiments/data_prep_comparison/data_prep_test_compare.csv`

### Suggested Order (for reproducible reporting)
1. `run_json_validity_f1_experiments.sh`
2. `run_format_comparison_and_csv.sh`
3. `run_generation_mode_comparison_and_csv.sh`
4. `run_data_prep_comparison_and_csv.sh`

---

## 🧪 Project 3 — Detailed Task Reports

### Task 1: Constrained Decoding for Structured NER

**Objective:** Train a LoRA adapter that outputs strict JSON with character-level offsets and a confidence score, then benchmark constrained vs free decoding on quality, JSON validity, and latency.

**Methodology:**

The pipeline consists of five stages, each implemented as a standalone module under `src/tasks/task1_constrained/`:

1. **Dataset preparation** (`prepare_dataset.py`): Loads the local CoNLL-2003 dataset, maps each example to a `{prompt, output}` JSONL format. The output is a JSON object containing an `entities` array (each with `type`, `value`, `start`, `end` character offsets) and a `confidence` score. Supports `--prompt_style with_defs` or `no_defs`.

2. **LoRA training** (`train.py`): Fine-tunes `Qwen/Qwen2.5-1.5B` with LoRA (`r=16`, `alpha=32`, `dropout=0.05`, targeting `q_proj` and `v_proj`). Training labels mask the prompt tokens with `-100` so loss is computed only on the JSON output. Default: 2 epochs, lr `2e-4`, batch size 1 with gradient accumulation 2.

3. **Constrained decoding** (`decode.py`): Uses the [Outlines](https://github.com/outlines-dev/outlines) library to build a JSON-schema logits processor from the NER schema (defined in `src/core/schema.py`). The schema enforces: top-level `entities` array + `confidence` float in [0,1], with each entity requiring `type`, `value`, `start` (int ≥ 0), `end` (int ≥ 0). The processor is attached to HuggingFace `model.generate()` (greedy: `do_sample=False`, `num_beams=1`, `max_new_tokens=256`, `repetition_penalty=1.05`). Free decoding uses the same generation config without the schema processor.

4. **Inference** (`inference.py`): Loads the base model + PEFT adapter, dispatches to constrained or free decoding per `--generation_mode`. With `--json_validate yes`, invalid JSON falls back to `{"entities": [], "confidence": 0.0}`.

5. **Benchmarking** (`benchmark.py`): Runs both modes on the test set, times wall-clock latency, and computes span-level metrics (entity sets compared as `(type, value, start, end)` tuples).

**Results:**

Source: `experiments/task1_constrained/task1_benchmark.csv`, `results.csv`

| Mode | Samples | Precision | Recall | F1 | Validity | Elapsed |
|------|---------|-----------|--------|----|----------|---------|
| Free | 500 | 0.5146 | 0.4486 | 0.4793 | 1.0 | 709.4 s |
| Constrained | 500 | 0.5146 | 0.4486 | 0.4793 | 1.0 | 1171.3 s |
| Free | 200 | 0.5770 | 0.5061 | 0.5393 | 1.0 | 288.1 s |
| Constrained | 200 | 0.5770 | 0.5061 | 0.5393 | 1.0 | 382.7 s |

**Key Findings:**

- At deterministic (temperature=0) decoding, free and constrained modes produce **identical** token sequences, resulting in the same precision, recall, and F1. The schema processor never needs to intervene because the LoRA-tuned model already generates valid JSON naturally.
- Both modes achieve **100% JSON validity** — the LoRA training effectively teaches the model the output schema.
- Constrained decoding adds **~65% latency overhead** (709 s → 1171 s for 500 samples) due to the Outlines logits processor evaluating schema constraints at every token.
- **Takeaway:** For a well-tuned model with deterministic decoding, the constrained decoder acts as a safety net but does not improve quality. It becomes valuable when using stochastic sampling (temperature > 0) where the model may occasionally deviate from the schema.

---

### Task 2: Layer-wise Entity Type Importance

**Objective:** Identify which transformer layers are most critical for recognizing each entity type (PER, ORG, LOC, MISC), using logit-lens analysis and per-layer LoRA ablation, then evaluate whether training LoRA on only the critical layers can match full-adapter performance.

**Methodology:**

Six modules under `src/tasks/task2_layer_importance/`:

1. **Logit lens** (`logit_lens.py`): For each of the 28 Qwen layers (0–27), projects the last-token hidden state through `lm_head` → softmax, and records the max probability over token IDs corresponding to entity type labels (`PER`, `ORG`, `LOC`, `MISC`). Computes an **emergence score** = mean probability when the label is present in ground truth minus mean probability when absent.

2. **Early vs late analysis** (`early_late_analysis.py`): Splits layers into early (0–13) and late (14–27), compares mean emergence scores per entity type to determine where each type's signal develops.

3. **Ablation** (`ablation.py`): Starting from the full Task 1 adapter, disables LoRA on `q_proj`/`v_proj` one layer at a time, runs inference, and measures the F1 drop (`f1_delta`) per entity type. Layers causing the largest drop are the most critical.

4. **Critical layer extraction** (`extract_critical_layers.py`): Takes the top-5 layers with the most negative `f1_delta` per entity type from the ablation results.

5. **Selective LoRA training** (`train_selective_lora.py`): Trains a new LoRA adapter using `layers_to_transform` restricted to the union of all critical layers (14 unique layers out of 28), with the same hyperparameters as full training.

6. **Summarization** (`summarize.py`): Aggregates logit-lens results into per-layer emergence scores.

**Critical Layers by Entity Type:**

Source: `experiments/task2_layer_importance/critical_layers.json`

| Entity Type | Critical Layers (top-5 by ablation F1 drop) |
|-------------|----------------------------------------------|
| ALL (macro) | 11, 12, 20, 6, 22 |
| PER | 11, 12, 16, 17, 9 |
| ORG | 12, 3, 4, 6, 11 |
| LOC | 11, 8, 16, 18, 20 |
| MISC | 20, 1, 25, 4, 6 |

**Early vs Late Layer Emergence:**

Source: `experiments/task2_layer_importance/results_early_late.csv`

| Entity Type | Stronger Half | Interpretation |
|-------------|--------------|----------------|
| ORG | Early (0–13) | Organization features develop in lower layers |
| MISC | Early (0–13) | Miscellaneous entity signals emerge early |
| LOC | Late (14–27) | Location recognition relies on higher-level representations |
| PER | Late (14–27) | Person name detection peaks in upper layers |

**Selective vs Full LoRA Comparison:**

Source: `experiments/task2_layer_importance/results_selective.csv` (evaluated on 200 validation examples)

| Adapter | Precision | Recall | F1 | Validity | Layers Trained |
|---------|-----------|--------|----|----------|----------------|
| Full (all 28 layers) | 0.3952 | 0.3791 | 0.3870 | 1.0 | 28 |
| Selective (critical only) | 0.3690 | 0.3511 | 0.3598 | 1.0 | 14 |

**Key Findings:**

- **Layers 11–12 are universally important** — they appear in the critical sets for ALL, PER, ORG, and are among the top layers overall. Ablating either causes significant F1 drops across all entity types.
- **Entity types have distinct layer dependencies**: ORG relies on early-mid layers (3, 4, 6), PER on mid-upper layers (9, 11, 12, 16, 17), LOC spans mid-to-late (8, 16, 18, 20), and MISC shows a bimodal pattern with both very early (1) and very late (25) layers.
- **Selective LoRA preserves 93% of full-adapter F1** (0.3598 vs 0.3870) while training only **half the layers** (14 out of 28), maintaining 100% JSON validity. This demonstrates that critical-layer-only training is a viable efficiency strategy, though the F1 gap suggests non-critical layers still contribute meaningful features.
- Emergence magnitudes are small (1e-6 to 1e-8 scale), so early/late conclusions are **directional indicators** rather than strong effect sizes.

---

### Task 3: Boundary Steering via Activation Intervention

**Objective:** Use activation steering to control whether the model produces strict or loose entity boundaries. Extract hidden-state activations from a separate model (`google/gemma-2-2b`), compute steering vectors representing the "strict boundary" direction, and apply them during inference to evaluate boundary F1 impact.

**Why Gemma 2B:** This task intentionally uses `google/gemma-2-2b` (a different architecture from the Qwen base used in Tasks 1/2/4/5) to study whether activation-space boundary signals are exploitable in a model that was **not** fine-tuned for the NER task.

**Methodology:**

Five-stage pipeline under `src/tasks/task3_steering/`:

1. **Boundary set preparation** (`prepare_boundary_sets.py`): From `task1_val.jsonl` (300 samples), creates two parallel datasets:
   - **Strict:** original prompt + gold JSON with exact entity spans.
   - **Loose:** same prompt + JSON where each entity boundary is randomly expanded or contracted by 1 character, with `value` re-sliced from the text accordingly.

2. **Activation extraction** (`extract_activations.py`): Loads `google/gemma-2-2b`, runs forward passes on both strict and loose examples with `output_hidden_states=True`. For each target layer (12–16), computes the **mean hidden state across all sequence positions**, then averages across all examples. Saves per-layer mean activations for both sets.

3. **Steering vector computation** (`compute_steering.py`): For each layer, computes `steering_vector = mean_strict - mean_loose` (elementwise subtraction). This vector points in the direction of "stricter boundaries" in activation space.

4. **Steered inference** (`run_steering.py`): Registers forward hooks on Gemma's decoder layers that add `scale * steering_vector` to the layer output during generation. Tests all combinations of 5 layers (12–16) × 3 scales (0.5, 1.0, 1.5) = 15 configurations on 200 test examples. Generation uses greedy decoding (`max_new_tokens=256`, `do_sample=False`).

5. **Boundary evaluation** (`evaluate_boundaries.py`): Computes span-level precision/recall/F1 by comparing `(start, end)` tuples between ground truth and predictions (ignoring entity type).

**Results:**

Source: `experiments/task3_steering/results.csv` (15 configurations)

| Layer | Scale | Precision | Recall | Boundary F1 |
|-------|-------|-----------|--------|-------------|
| 12 | 0.5 | 0.00351 | 0.00491 | 0.00409 |
| 12 | 1.0 | 0.00348 | 0.00491 | 0.00408 |
| 12 | 1.5 | 0.00348 | 0.00491 | 0.00408 |
| **13** | **0.5** | **0.00350** | **0.00491** | **0.00409** |
| **13** | **1.0** | **0.00355** | **0.00491** | **0.00412** |
| **13** | **1.5** | **0.00357** | **0.00491** | **0.00413** |
| 14 | 0.5 | 0.00350 | 0.00491 | 0.00409 |
| 14 | 1.0 | 0.00350 | 0.00491 | 0.00409 |
| 14 | 1.5 | 0.00350 | 0.00491 | 0.00409 |
| 15 | 0.5 | 0.00347 | 0.00491 | 0.00407 |
| 15 | 1.0 | 0.00351 | 0.00491 | 0.00410 |
| 15 | 1.5 | 0.00348 | 0.00491 | 0.00407 |
| 16 | 0.5 | 0.00347 | 0.00491 | 0.00407 |
| 16 | 1.0 | 0.00348 | 0.00491 | 0.00407 |
| 16 | 1.5 | 0.00348 | 0.00491 | 0.00408 |

Best configuration: **Layer 13, Scale 1.5** → Boundary F1 **0.00413**

**Key Findings:**

- **Recall is constant across all configurations** (0.00491), meaning the same number of gold spans are matched regardless of layer/scale. Steering primarily affects **precision** (false positive rate), not the model's ability to detect correct spans.
- The **best F1 (Layer 13, Scale 1.5)** is only marginally better than the worst (Layer 15/16, Scale 0.5), with differences on the order of **0.00006** — too small to be practically meaningful.
- **Overall F1 is extremely low (~0.4%)**, reflecting that Gemma 2B without task-specific fine-tuning cannot reliably produce structured NER JSON. The steering intervention is working against a very weak baseline.
- **Train/inference mismatch:** Steering vectors were computed from activations on `prompt + gold output` (full sequence), but during inference the model only sees the prompt and must generate the output — this distribution shift likely weakens the steering effect.
- **Layer 13 shows the strongest (though still modest) response** to steering, consistent with mid-layer representations being most amenable to directional control.
- **Future direction:** Scales 2.0–3.0 could probe stronger effects, and applying steering to a task-tuned model (rather than a base model) would provide a more meaningful baseline.

---

### Task 4: Adversarial Robustness

**Objective:** Evaluate and improve the model's robustness against five categories of adversarial entity perturbations by training on a mixed dataset of original + adversarial examples.

**Methodology:**

Seven modules under `src/tasks/task4_adversarial/`:

1. **Adversarial example generation** (`prepare_eval_set.py`, `prepare_train_set.py`): Categories are assigned round-robin. Each perturbation is applied via `_transform_row`:

   | Category | Perturbation Strategy |
   |----------|----------------------|
   | **Nested** | Prefixes `"International "` to the first entity, creates an outer entity span wrapping the expanded text, producing overlapping/nested annotations |
   | **Abbrev** | Replaces multi-word entity values with initials (e.g., "New York" → "NY"); single words ≥4 chars → first 3 chars uppercased (e.g., "LONDON" → "LON"). Falls back to misspell if abbreviation fails |
   | **Misspell** | Swaps two middle characters in entity values ≥4 chars (e.g., "West Indian" → "WsetIndian"). Falls back to multilingual on failure |
   | **Ambiguous** | Wraps entity value in double quotes within the text while keeping the entity span pointing to the unquoted value, testing whether surrounding punctuation confuses span detection |
   | **Multilingual** | Appends `" Texto adicional en espanol."` to the text, testing whether foreign-language noise degrades entity extraction on the English portion |

2. **Training** (`train.py`): Creates `train_mixed.jsonl` (full original training set + 1000 adversarial examples), fine-tunes with LoRA (same config as Task 1: `r=16`, `alpha=32`, `q_proj`/`v_proj`, 2 epochs).

3. **Evaluation** (`evaluate.py`): Computes per-category precision/recall/F1 at both entity-level (matching `(type, value, start, end)` tuples) and boundary-level (matching `(start, end)` only). Also tracks JSON validity.

4. **Comparison** (`compare_results.py`): Computes `post - pre` metric differences per category → `robustness_gains.csv`.

**Pre- vs Post-Training Results:**

Source: `experiments/task4_adversarial/results_pre.csv`, `results_post.csv`, `robustness_gains.csv`

| Category | Pre F1 | Post F1 | F1 Gain | Pre Boundary F1 | Post Boundary F1 | Boundary F1 Gain |
|----------|--------|---------|---------|-----------------|-------------------|-----------------|
| Nested | 0.0889 | 0.3333 | **+0.2444** | 0.0889 | 0.3969 | +0.3079 |
| Abbrev | 0.2462 | 0.3235 | **+0.0774** | 0.2769 | 0.4412 | +0.1643 |
| Misspell | 0.3333 | 0.4359 | **+0.1026** | 0.3333 | 0.5897 | +0.2564 |
| Ambiguous | 0.1235 | 0.4048 | **+0.2813** | 0.1235 | 0.4048 | +0.2813 |
| Multilingual | 0.0992 | 0.6486 | **+0.5495** | 0.1322 | 0.6757 | +0.5434 |
| Original | 0.2916 | 0.5052 | +0.2136 | 0.3274 | 0.5103 | +0.1829 |
| All | 0.2267 | 0.4597 | +0.2329 | 0.2506 | 0.4988 | +0.2482 |
| Adversarial (all) | 0.1700 | 0.4186 | **+0.2486** | 0.1834 | 0.4884 | +0.3049 |

**Key Findings:**

- **Multilingual perturbations saw the largest improvement** (+0.5495 F1), going from near-random performance (0.099) to strong extraction (0.649). The model learned to ignore appended foreign-language noise.
- **Ambiguous and nested categories also improved dramatically** (+0.28 and +0.24 F1 respectively), showing the model became more resilient to surrounding punctuation and overlapping span structures.
- **Original (clean) data performance also improved** (+0.21 F1), indicating that adversarial training acts as a form of data augmentation that benefits even non-adversarial inputs.
- **JSON validity remained 100%** across all categories in both pre and post evaluations — the adversarial training did not compromise output format compliance.
- **Boundary F1 consistently exceeds entity F1**, meaning the model finds correct span boundaries more often than it gets the full `(type, value, start, end)` tuple right — entity type classification is the harder sub-problem.
- **Note:** Pre-evaluation uses `eval_combined.jsonl` (from val split) while post-evaluation uses `heldout_combined.jsonl` (from test split), so gains reflect both model improvement and dataset differences.

---

### Task 5: Production Profiling & Quantization

**Objective:** Profile inference memory usage, benchmark three quantization levels (Q4\_K\_M, Q5\_K\_M, Q8\_0) for latency/quality tradeoffs, measure concurrency scaling behavior, and recommend a production deployment configuration that meets a p95 < 500ms SLA target.

**Methodology:**

Three modules under `src/tasks/task5_production/`:

1. **Memory profiling** (`profile_memory.py`): Loads the full `Qwen/Qwen2.5-1.5B` model in HuggingFace, runs forward passes on 50 test examples with `output_hidden_states=True`, and measures per-layer activation tensor sizes (`numel * element_size`) and total model parameter memory.

2. **Quantization benchmarking** (`benchmark_llamacpp.py`): Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) to run inference with quantized GGUF models. The quantization pipeline is:
   - Convert HF model → GGUF FP16 via `convert_hf_to_gguf.py`
   - Quantize FP16 → Q4\_K\_M / Q5\_K\_M / Q8\_0 via `llama.cpp/quantize`
   - Benchmark each variant on the full CoNLL-2003 test set (3453 samples) with deterministic decoding (`temp=0`, `seed=42`)

3. **Concurrency benchmarking** (`concurrency_benchmark.py`): Uses `ThreadPoolExecutor` to simulate concurrent requests at levels 1/2/4/8/16, measuring per-request latency distributions (p50/p95/p99) and throughput. Tests both subprocess-spawn and HTTP server modes.

**Memory Profile:**

Source: `experiments/task5_production/memory_profile.csv`, `memory_profile.json`

| Component | Memory |
|-----------|--------|
| Model parameters (FP16) | **2944.4 MB** (3,087,428,608 bytes) |
| Activation memory per layer (mean) | **0.33 MB** (346,460 bytes) |
| Activation total (27 layers) | **~8.9 MB** |

**Conclusion:** The memory bottleneck is overwhelmingly **model weights** (~2.9 GB), not activations (~9 MB). Quantization directly addresses this by reducing weight storage by 47–68%.

**Quantization Benchmark (full CoNLL-2003 test set, n=3453):**

Source: `experiments/task5_production/quant_benchmark_with_memory.csv`

| Quantization | File Size | Per-sample Latency | Precision | Recall | F1 | Validity |
|-------------|-----------|-------------------|-----------|--------|----|----------|
| Q4\_K\_M | 940 MB | 1922.7 ms | 0.3665 | 0.3484 | 0.3573 | 1.0 |
| Q5\_K\_M | 1073 MB | 2646.7 ms | 0.4201 | 0.4040 | 0.4119 | 1.0 |
| Q8\_0 | 1570 MB | 2108.3 ms | 0.4565 | 0.4391 | 0.4476 | 1.0 |

**Quality-size tradeoff:** Q8\_0 achieves the highest F1 (0.4476) at 1.57 GB; Q4\_K\_M is 40% smaller (940 MB) but loses ~9 F1 points. Q5\_K\_M is the slowest despite being mid-sized.

**Concurrency Scaling (Q8\_0, Metal acceleration, n=32):**

Source: `experiments/task5_production/concurrency.csv`

| Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) |
|-------------|----------|----------|----------|--------------------|
| 1 | 566.8 | 579.9 | 632.7 | 1.76 |
| 4 | 575.0 | 613.2 | 619.7 | 6.88 |
| 8 | 633.1 | 642.1 | 645.0 | 12.62 |
| 16 | 708.8 | 752.5 | 1198.0 | 21.65 |

**Scaling observations:** Throughput scales nearly linearly up to c=8 (~12.6 req/s), but c=16 introduces severe tail latency (p99 jumps to ~1.2s) while throughput gains diminish.

**SLA-Tuned Server Runs (max\_tokens=64, n=64):**

| Config | Concurrency | p50 (ms) | p95 (ms) | Throughput (req/s) |
|--------|-------------|----------|----------|--------------------|
| Q4\_K\_M tuned | 1 | 268.0 | **283.7** | 3.96 |
| Q4\_K\_M tuned | 2 | 372.6 | **427.0** | 5.65 |
| Q5\_K\_M tuned | 1 | 348.6 | 363.7 | 3.05 |
| Q5\_K\_M tuned | 2 | 525.8 | 583.9 | 4.01 |
| Q8\_0 server | 1 | 634.3 | 662.4 | 1.83 |

**Production Recommendations:**

Source: `experiments/task5_production/production_recommendation.md`

| Profile | Quantization | Batch Size | Concurrency | p95 Latency | Throughput | F1 |
|---------|-------------|------------|-------------|-------------|------------|----|
| **Quality-first** | Q8\_0 | 1 | 1 | 662.4 ms | 1.83 req/s | 0.4476 |
| **SLA-first** (< 500ms p95) | Q4\_K\_M | 1 | 2 | **427.0 ms** | 5.65 req/s | 0.3573 |

**Key Findings:**

- **SLA target (p95 < 500ms) is met** with the tuned Q4\_K\_M configuration: server-style inference, `max_tokens=64`, batch size 1, concurrency 2 → p95 = 427ms at 5.65 req/s.
- **Q8\_0 provides the best quality** (F1 0.4476) but cannot meet the 500ms SLA without further optimization.
- **Model file size directly correlates with accuracy:** Q4 (940 MB) → F1 0.36, Q5 (1073 MB) → F1 0.41, Q8 (1570 MB) → F1 0.45. This represents a **68% size reduction** from FP16 (2944 MB) to Q4 with a manageable quality trade-off.
- **Concurrency beyond 8 degrades tail latency** disproportionately — p99 at c=16 is nearly 2x that of c=8, making high concurrency unsuitable for strict SLA deployments.
- **Reducing `max_tokens` from 256 to 64** was the single most impactful latency optimization, cutting per-request time by more than half in server mode.

---

## 🧰 Project3 Task 5 Reproducibility Notes (llama.cpp + LFS)

### Why `llama.cpp`
`llama.cpp` is used in Task 5 to convert/quantize GGUF models and run low-level quantized inference benchmarks.

### Install/build `llama.cpp` (for new users)
If `cmake` is missing (`command not found: cmake`), install it first:
```bash
# macOS (Homebrew)
brew install cmake
cmake --version
```

For Linux, install `cmake` from your package manager before building.

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -S . -B build -DGGML_METAL=ON
cmake --build build -j
```

### Git LFS setup (for model artifacts)
```bash
git lfs install
git lfs track "models/**/*.gguf"
git lfs track "models/**/*.safetensors"
git lfs track "models/**/*.bin"
git add .gitattributes
```

### Pull LFS files after clone
```bash
git lfs pull
```

---

## 🖥️ Gradio UI (Inference + Correction + Active Learning)
Main UI file: `src/gradio_correction_app.py`

### Includes
- **Extract tab**: text -> structured JSON + highlighted entities
- **Correct & Save tab**: editable JSON + approval/save
- **Analytics Dashboard tab**: embedded interactive Plotly dashboard

### Output artifacts from UI
- Predictions: `data/processed/active_learning/predictions_export.jsonl`
- Corrections: `data/processed/corrections/corrections.jsonl`
- AL cycles: `data/processed/active_learning/cycle_records.jsonl`

### Launch
```bash
python -m src.gradio_correction_app
```

Public temporary link:
```bash
python -m src.gradio_correction_app --share
```

---

## 📈 Plot Dashboard

Generate/rebuild dashboard:
```bash
python scripts/build_plotly_dashboard.py
```

Open locally:
```bash
open plots/index.html
```

---

## 🎯 Recommended Production Config

Use this for stable deployment:
- model/adaptor: `experiments/with_defs_qwen2_5_1_5B`
- json_validate: `yes`
- temperature: `0.1`
- output format: `json`
- generation mode: `constrained`

---

## 📌 Future Improvements
- Add per-sample scoring in UI (when ground truth provided)
- Add model switcher in Gradio (baseline vs with_defs)
- Add review-queue prioritization by uncertainty
- Add hosted demo deployment pipeline

---

## ✨ Author
Devolped by Krishna Chaitanya.
