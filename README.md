# LLM Fine-Tuning: SFT and DPO on TinyLlama-1.1B

An end-to-end NLP assignment implementing supervised fine-tuning (SFT) followed by direct preference optimisation (DPO) on **TinyLlama-1.1B**, run entirely on Google Colab's free T4 GPU. Five trials are run for each phase, with the best adapter selected by BERTScore F1 and carried forward to the next phase.

---

## Results Summary

### Main Metrics — Base vs Best SFT vs Best DPO

| Model | Configuration | Mean BLEU | Mean BERTScore F1 |
|-------|--------------|-----------|-------------------|
| Base TinyLlama-1.1B | No fine-tuning | 2.60 | −0.016 |
| **Best SFT — T2** | r=8, q+v, lr=2e-4, 2 epochs | **6.45** | **0.222** |
| Best DPO — D3 | beta=0.3, lr=1e-5, 1 epoch | 2.998 | 0.051 |

SFT (T2) produced a large improvement on both metrics. DPO (D3) improves over the base model but regresses against the SFT baseline — primarily due to distributional shift between the Orca-format training pairs and the Alpaca-format evaluation prompts.

### All SFT Trials

| Trial | r | Target Modules | LR | Epochs | Val Loss | BLEU | BERTScore F1 |
|-------|---|----------------|----|--------|----------|------|--------------|
| T1 | 4 | q, v | 2e-4 | 1 | 1.0083 | 4.88 | 0.1889 |
| **T2** | **8** | **q, v** | **2e-4** | **2** | **0.9867** | **6.45** | **0.2210** |
| T3 | 8 | q, v, o | 1e-4 | 2 | 0.9819 | 5.27 | 0.2155 |
| T4 | 16 | q, k, v, o, gate, up, down | 5e-5 | 1 | 0.9801 | 5.20 | 0.2198 |
| T5 | 8 | q, v | 5e-4 | 1 | 0.9873 | 5.69 | 0.1872 |

### All DPO Trials

| Trial | Beta | LR | Epochs | Val Loss | BLEU | BERTScore F1 |
|-------|------|----|--------|----------|------|--------------|
| D1 | 0.1 | 1e-5 | 1 | 0.3529 | 3.208 | −0.007 |
| D2 | 0.1 | 5e-5 | 2 | 0.0316 | 2.305 | −0.008 |
| **D3** | **0.3** | **1e-5** | **1** | **0.1789** | **2.998** | **0.035** |
| D4 | 0.5 | 2e-5 | 2 | 0.0341 | 2.814 | 0.006 |
| D5 | 0.1 | 1e-4 | 1 | 0.0375 | 3.369 | −0.004 |

---

## Sample Outputs

### Prompt 1 — Factual: *"What is the greenhouse effect and how does it contribute to climate change?"*

| Model | Output (excerpt) | BERTScore F1 |
|-------|-----------------|--------------|
| **Base** | *The greenhouse effect is the phenomenon whereby certain gases absorb and trap the infrared radiation emitted by the earth… [then generates more ### Instruction: blocks]* | 0.104 |
| **SFT T2** | *The greenhouse effect is the phenomenon that occurs when gases in the atmosphere trap the heat that is emitted from the Earth's surface. This keeps the Earth's surface temperature at a warmer level…* | 0.258 |
| **DPO D3** | *The greenhouse effect is the effect that greenhouse gases have on the earth's atmosphere. The greenhouse effect is the phenomenon that traps heat that is radiated from the earth's surface…* | 0.086 |

### Prompt 4 — Instruction: *"List five practical tips for improving productivity while working from home."*

| Model | Output (excerpt) | BERTScore F1 |
|-------|-----------------|--------------|
| **Base** | *- Set a timer / - Use a timer on your phone / - Use a timer on your computer / - Use a timer on your computer at the beginning of every day [repeated 10+ times]* | −0.206 |
| **SFT T2** | *1. Set a reasonable work schedule. 2. Set up a dedicated workspace and ensure that all equipment is in working order. 3. Take regular breaks…* | 0.322 |
| **DPO D3** | *1. I find it easier to work on my computer as opposed to a phone. 2. I work on my computer on a regular basis. 3. I try to…* | 0.029 |

---

## Project Structure

```
NLP Finetuning project/
│
├── 00_baseline_evaluation.ipynb   # Phase 1 — baseline eval of raw TinyLlama
├── 01_sft_training.ipynb          # Phase 2 — 5 SFT trials (training only)
├── 01b_sft_eval_results.ipynb     # Phase 2b — load saved SFT adapters, compute metrics
├── 02_dpo_training.ipynb          # Phase 3 — 5 DPO trials (training only)
├── 02b_dpo_eval_results.ipynb     # Phase 3b — load saved DPO adapters, compute metrics
├── 03_per_prompt_eval.ipynb       # Per-prompt BERTScore for Table 6 in report
│
├── report.md                      # Full assignment report (convert to .docx to submit)
├── nlp project plan.txt           # Original implementation plan
│
├── done/                          # All notebooks executed on Colab with full cell outputs
│   ├── 00_baseline_evaluation-completed.ipynb
│   ├── 01_sft_training.ipynb
│   ├── 01b_sft_eval_results.ipynb
│   ├── 02_dpo_training.ipynb
│   └── 02b_dpo_eval_results.ipynb
│
├── results/                       # CSV files with per-trial BLEU and BERTScore F1
│   ├── baseline_results.csv       # Per-prompt scores for the base model
│   ├── sft_results.csv            # All 5 SFT trials (T1–T5)
│   └── dpo_results.csv            # All 5 DPO trials (D1–D5)
│
└── doc/                           # Project session context and exported logs
```

---

## Setup Guide

### Requirements
- Google Colab with **T4 GPU** (Runtime → Change runtime type → T4 GPU)
- Google Drive mounted at `/content/drive`
- `test_prompts.json` with 10 prompts and GPT-4o reference answers uploaded to `MyDrive/Colab Notebooks/data/`

### Package Versions

**Baseline notebook (`00_baseline_evaluation.ipynb`):**
```bash
pip install transformers==4.52.4 bitsandbytes==0.46.0 accelerate==1.7.0 \
    triton==3.6.0 evaluate==0.4.3 bert-score==0.3.13 sacrebleu==2.4.3
```

**SFT notebooks (`01_sft_training.ipynb`, `01b_sft_eval_results.ipynb`):**
```bash
pip install numpy==1.26.4 transformers==4.52.4 trl==0.9.6 peft==0.12.0 \
    accelerate==1.7.0 bitsandbytes==0.46.0 triton==3.6.0 datasets==3.5.0 \
    evaluate==0.4.3 bert-score==0.3.13 sacrebleu==2.4.3
```

**DPO notebooks (`02_dpo_training.ipynb`, `02b_dpo_eval_results.ipynb`, `03_per_prompt_eval.ipynb`):**
```bash
pip install numpy==1.26.4 transformers==4.44.2 trl==0.9.6 peft==0.12.0 \
    accelerate==0.34.2 bitsandbytes==0.46.0 triton==3.6.0 datasets==2.21.0 \
    evaluate==0.4.3 bert-score==0.3.13 sacrebleu==2.4.3
```

> **Note:** DPO notebooks use `transformers==4.44.2` — `trl==0.9.6`'s `DPOTrainer` is incompatible with transformers >= 4.45 (`get_batch_samples` signature mismatch). SFT notebooks are unaffected.

### Execution Order

1. `00_baseline_evaluation.ipynb` — produces `baseline_results.csv`
2. `01_sft_training.ipynb` — run in batches if needed (T1–T2, T3, T4–T5); adapters saved to `MyDrive/sft_trials/`
3. `01b_sft_eval_results.ipynb` — loads all adapters, produces `sft_results.csv`; note best trial (T2)
4. Update `BEST_SFT_ADAPTER_PATH = "/content/drive/MyDrive/sft_trials/T2"` in `02_dpo_training.ipynb` Cell 4
5. `02_dpo_training.ipynb` — training only; adapters saved to `MyDrive/dpo_trials/`
6. `02b_dpo_eval_results.ipynb` — loads all DPO adapters, produces `dpo_results.csv` and three-way comparison

### Key Google Drive Paths
```
MyDrive/
├── sft_trials/T1 ... T5     # SFT adapter folders
├── dpo_trials/D1 ... D5     # DPO adapter folders
├── baseline_results.csv
├── sft_results.csv
├── dpo_results.csv
└── Colab Notebooks/data/test_prompts.json
```

---

## Model Details

| Component | Value |
|-----------|-------|
| Base model | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| SFT dataset | `tatsu-lab/alpaca` — 10,000-sample subset |
| DPO dataset | `Intel/orca_dpo_pairs` — 3,000-sample subset |
| Fine-tuning method | LoRA (PEFT) via TRL |
| Quantization | 4-bit NF4 (bitsandbytes) |
| Evaluation | SacreBLEU + BERTScore F1 (rescaled, roberta-large) |
| Best SFT adapter | `MyDrive/sft_trials/T2` |
| Best DPO adapter | `MyDrive/dpo_trials/D3` |
