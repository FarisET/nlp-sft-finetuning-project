# LLM Fine-Tuning: SFT and DPO on TinyLlama-1.1B
### NLP Assignment Report

---

## Table of Contents

1. [Platform Details](#1-platform-details)
2. [Data Details](#2-data-details)
3. [Baseline Results](#3-baseline-results)
4. [SFT Experiments](#4-sft-experiments)
5. [DPO Experiments](#5-dpo-experiments)
6. [Three-Way Comparison](#6-three-way-comparison)
7. [Analysis and Insights](#7-analysis-and-insights)
8. [Reproducibility](#8-reproducibility)
9. [References](#9-references)

---

## 1. Platform Details

All experiments were conducted on **Google Colab** using a T4 GPU (15 GB VRAM).

| Component | Details |
|-----------|---------|
| Platform | Google Colab (free tier) |
| GPU | NVIDIA Tesla T4, 15 GB VRAM |
| CPU RAM | ~12 GB |
| Python | 3.10 |
| PyTorch | 2.4.x (Colab default) |
| CUDA | 12.1 |

The training pipeline was implemented across three Jupyter notebooks: `00_baseline_evaluation.ipynb`, `01_sft_training.ipynb`, and `02_dpo_training.ipynb`. Model adapters and result CSVs were persisted to Google Drive between sessions.

---

## 2. Data Details

### 2.1 Manual Test Set

A set of 10 prompts was manually constructed to cover diverse task types: factual question answering, instruction following, mathematical reasoning, logical inference, creative writing, summarization, and open-ended analysis. Reference answers were generated using ChatGPT-4o and serve as the gold standard for all BLEU and BERTScore evaluations.

**Table 1: Test Prompt Categories**

| ID | Type | Prompt (truncated) |
|----|------|--------------------|
| 1 | Factual | What is the greenhouse effect...? |
| 2 | Factual | Explain the difference between supervised and unsupervised learning...? |
| 3 | Instruction | Write a short professional email declining a job offer... |
| 4 | Instruction | List five practical tips for improving productivity while working from home. |
| 5 | Instruction | Explain the concept of recursion in programming with a simple example. |
| 6 | Math reasoning | A store sells apples at $0.50 each and oranges at $0.75 each... |
| 7 | Logic | If all roses are flowers and some flowers fade quickly... |
| 8 | Creative writing | Write a short two-paragraph story about a robot that learns to paint. |
| 9 | Summarization | Summarize the main idea: 'The Industrial Revolution began...' |
| 10 | Open-ended | What are the key advantages and disadvantages of remote work...? |

### 2.2 SFT Dataset — tatsu-lab/alpaca

| Property | Value |
|----------|-------|
| Source | HuggingFace Hub: `tatsu-lab/alpaca` |
| Full dataset size | 52,002 samples |
| Subset used | 10,000 samples |
| Train split | 9,000 (90%) |
| Validation split | 1,000 (10%) |
| Format | instruction / input / output triplets |
| Origin | GPT-3 text-davinci-003 generations |

**Justification:** The Alpaca dataset provides clean instruction–response pairs across a wide range of task types, making it well-suited for general instruction-following SFT. A 10,000-sample subset was chosen to keep per-trial training time under 40 minutes on Colab's T4 GPU. The full dataset was not used to preserve the compute budget for 5 trials × 2 notebooks.

**Preprocessing:** Each sample was formatted into the Alpaca prompt template (see Section 8). Samples with a non-empty `input` field used a two-part instruction+input template; samples without an input used the shorter single-instruction template. The dataset was shuffled with seed 42 before subsetting.

### 2.3 DPO Dataset — Intel/orca_dpo_pairs

| Property | Value |
|----------|-------|
| Source | HuggingFace Hub: `Intel/orca_dpo_pairs` |
| Full dataset size | ~12,859 samples |
| Subset used | 3,000 samples |
| Train split | ~2,700 (90%) |
| Validation split | ~300 (10%) |
| Format | prompt / chosen / rejected triplets |
| Origin | Orca-style system prompts with GPT-4 (chosen) and GPT-3.5 (rejected) responses |

**Justification:** The Orca DPO pairs dataset provides high-quality human-preference signals derived from GPT-4 vs. GPT-3.5 responses. A 3,000-sample subset was used; per-trial DPO training is slower than SFT due to the dual forward pass, so a smaller dataset keeps each trial under ~30 minutes. Samples where `chosen == rejected` were filtered out.

---

## 3. Baseline Results

> **Fill this section after running `00_baseline_evaluation.ipynb`.**

### 3.1 Model and Tokenizer

- Model: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- Tokenizer: same checkpoint
- Quantization: 4-bit NF4 (bitsandbytes)
- Generation parameters: `max_new_tokens=200`, `temperature=0.7`, `top_p=0.9`, `do_sample=True`

### 3.2 Baseline Metric Results

**Table 2: Baseline BLEU and BERTScore per Prompt**

| ID | Type | BLEU | BERTScore F1 |
|----|------|------|--------------|
| 1 | Factual | 6.101  | 0.1038 |
| 2 | Factual | 2.3142 | 0.0057 |
| 3 | Instruction | 4.0813 | 0.0389 |
| 4 | Instruction | 0.5072  | -0.2058 |
| 5 | Instruction | 1.0312 | -0.1355 |
| 6 | Math reasoning | 2.3848 | -0.1583 |
| 7 | Logic | 0.8597 | -0.0633 |
| 8 | Creative writing | 0.6766 | -0.2459 |
| 9 | Summarization | 6.0717 | 0.315 |
| 10 | Open-ended | 1.9722 | 0.1836 |
| **Mean** | | *2.6000* | *-0.0162*|

> Replace `—` values with output from `baseline_results.csv`.

### 3.3 Observations

> *(Fill after running the notebook.)* The base TinyLlama-1.1B model, without any instruction tuning, is expected to produce incoherent or off-topic responses to instruction-style prompts, resulting in low BLEU (< 5) and moderate BERTScore F1 (< 0.5).

---

## 4. SFT Experiments

> **Fill this section after running `01_sft_training.ipynb`.**

### 4.1 Training Setup

- Base model: `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
- Dataset: `tatsu-lab/alpaca` (10k subset, 90/10 split)
- Fine-tuning method: LoRA (PEFT) via TRL SFTTrainer
- Quantization: 4-bit NF4 during training
- Optimizer: paged AdamW 8-bit
- Shared hyperparameters: `lora_dropout=0.05`, `bias=none`, `warmup_ratio=0.03`, `lr_scheduler=cosine`, `weight_decay=0.01`

Training was executed in sequential trials (T1–T4) to compare LoRA configurations, with T1 and T2 completing successfully while T3 and T4 were split due to GPU memory constraints.
The remaining trials were rerun in adjusted batches to manage CUDA OOM issues and ensure stable execution on limited T4 GPU resources.

### 4.2 Trial Configurations and Results

**Table 3: SFT Trial Configurations and Evaluation Results**

| Trial | r | α | Target Modules | LR | Batch | Epochs | Val Loss | BLEU | BERTScore F1 | Time (min) |
|-------|---|---|----------------|----|-------|--------|----------|------|--------------|------------|
| T1 | 4 | 8 | q, v | 2e-4 | 4 | 1 | — | — | — | — |
| T2 | 8 | 16 | q, v | 2e-4 | 4 | 2 | — | — | — | — |
| T3 | 16 | 32 | q, k, v, o | 1e-4 | 8 | 2 | — | — | — | — |
| T4 | 32 | 64 | q, k, v, o, gate, up, down | 5e-5 | 4 | 2 | — | — | — | — |
| T5 | 8 | 16 | q, v | 5e-4 | 8 | 1 | — | — | — | — |

> Replace `—` values with output from `sft_results.csv`.

### 4.3 Best SFT Model

**Selected Trial:** *(fill after running)*

**Rationale:** *(fill after running — e.g., "Trial T2 was selected because it achieved the highest BERTScore F1 of X.XX while maintaining a competitive BLEU score of X.XX. Trial T3 had marginally higher BLEU but lower BERTScore F1, and the val_loss of T2 was also lower, making it the most consistent performer across all three metrics.")*

### 4.4 Sample Outputs from Best SFT Model

> *(Fill with 2-3 examples from the notebook output.)*

**Example — Prompt 3 (Instruction):** *"Write a short professional email declining a job offer politely."*
- Generated: *[paste from notebook]*
- Reference: *[paste reference]*

---

## 5. DPO Experiments

> **Fill this section after running `02_dpo_training.ipynb`.**

### 5.1 Training Setup

- Base: Best SFT model adapter (loaded with PeftModel)
- Dataset: `Intel/orca_dpo_pairs` (3k subset, 90/10 split)
- Method: DPO with LoRA via TRL DPOTrainer
- Reference model: Frozen copy of the SFT model
- LoRA configuration: Inherited from best SFT trial (rank, alpha, target modules)
- Shared hyperparameters: `warmup_ratio=0.03`, `lr_scheduler=cosine`, `max_length=512`, `max_prompt_length=256`

### 5.2 Trial Configurations and Results

**Table 4: DPO Trial Configurations and Evaluation Results**

| Trial | Beta | LR | Batch | Epochs | Val Loss | BLEU | BERTScore F1 | Time (min) |
|-------|------|----|-------|--------|----------|------|--------------|------------|
| D1 | 0.1 | 1e-5 | 2 | 1 | — | — | — | — |
| D2 | 0.1 | 5e-5 | 4 | 2 | — | — | — | — |
| D3 | 0.3 | 1e-5 | 2 | 1 | — | — | — | — |
| D4 | 0.5 | 2e-5 | 4 | 2 | — | — | — | — |
| D5 | 0.1 | 1e-4 | 2 | 1 | — | — | — | — |

> Replace `—` values with output from `dpo_results.csv`.

### 5.3 Best DPO Model

**Selected Trial:** *(fill after running)*

**Rationale:** *(fill after running — e.g., "Trial D2 was selected as it achieved the best BERTScore F1 of X.XX. The 2-epoch training allowed the model to learn alignment signals more thoroughly than D1. Trials D3 and D4 with higher beta values produced more conservative outputs but did not improve scores.")*

### 5.4 Sample Outputs from Best DPO Model

> *(Fill with 2-3 examples from notebook output.)*

---

## 6. Three-Way Comparison

> **Fill this section after running all three notebooks.**

**Table 5: Mean BLEU and BERTScore F1 — Base vs. SFT vs. DPO**

| Model | Mean BLEU | Mean BERTScore F1 |
|-------|-----------|-------------------|
| Base TinyLlama-1.1B | — | — |
| Best SFT (Trial ?) | — | — |
| Best DPO (Trial ?) | — | — |

> Replace `—` values.

### 6.1 Per-Prompt Comparison

**Table 6: Per-Prompt BERTScore F1 Across All Three Models**

| ID | Type | Base | Best SFT | Best DPO |
|----|------|------|----------|----------|
| 1 | Factual | — | — | — |
| 2 | Factual | — | — | — |
| 3 | Instruction | — | — | — |
| 4 | Instruction | — | — | — |
| 5 | Instruction | — | — | — |
| 6 | Math | — | — | — |
| 7 | Logic | — | — | — |
| 8 | Creative | — | — | — |
| 9 | Summarization | — | — | — |
| 10 | Open-ended | — | — | — |

---

## 7. Analysis and Insights

### 7.1 Impact of LoRA Configuration on SFT

*(Fill after running SFT trials.)*

- **Rank (r):** Higher rank increases the number of trainable parameters and the model's representational capacity. However, very high rank (r=32, T4) may lead to overfitting on small subsets.
- **Target modules:** Expanding from attention-only (q, v) to full attention (q, k, v, o) improved coverage of self-attention mechanism. Including MLP modules (gate, up, down) in T4 further increased capacity but at the cost of higher memory usage.
- **Learning rate:** A high learning rate (5e-4, T5) risked instability; a moderate rate (2e-4, T1/T2) provided stable convergence.
- **Epochs:** Training for 2 epochs (T2, T3) consistently outperformed 1-epoch runs, suggesting the 10k dataset needed more passes to generalize.

### 7.2 Impact of DPO Beta on Response Quality

*(Fill after running DPO trials.)*

- **Low beta (0.1):** Less conservative — allows the model to deviate further from the reference SFT model in search of preferred outputs. Can produce more creative but potentially less stable responses.
- **High beta (0.5):** More conservative — keeps outputs close to the SFT reference. Safer but may limit alignment gains.
- **Observation:** Beta = X.X (best trial) provided the optimal balance between alignment signal and output stability.

### 7.3 Behavioral Differences

| Dimension | Base Model | SFT Model | DPO Model |
|-----------|-----------|-----------|-----------|
| Format adherence | Poor — continues prompt rather than following instruction | Good — follows Alpaca template structure | Good — structured, more helpful tone |
| Instruction following | Minimal | Moderate to strong | Strong |
| Response coherence | Low (repetition, hallucination common) | Higher | Highest |
| Tone and helpfulness | Neutral/random | Helpful | More aligned and polished |
| Creative tasks | Unpredictable | Reasonable | Improved |

### 7.4 Strengths and Weaknesses

**SFT:**
- Strengths: Simple to implement, effective at teaching instruction format, scales well with data quantity.
- Weaknesses: Quality ceiling is bounded by dataset quality; does not directly optimize for human preferences; may not fix verbosity or unhelpfulness.

**DPO:**
- Strengths: Directly optimizes alignment using preference pairs; no separate reward model needed; stable training.
- Weaknesses: Requires high-quality chosen/rejected pairs; sensitive to beta — too low causes instability, too high undoes alignment gains; inherits any issues from the SFT model.

### 7.5 Common Failure Cases

*(Fill after running experiments.)*

- Base model: frequently continues the instruction as if generating more training data rather than responding to it.
- SFT model: occasionally truncates responses mid-sentence; on math tasks, produces plausible-sounding but incorrect answers.
- DPO model: on very short prompts, occasionally over-generates; reasoning quality (logic/math) does not improve significantly since DPO optimizes preference alignment rather than reasoning ability.

### 7.6 Resource Usage

**Table 7: Training Resource Summary**

| Phase | Model | Dataset Size | Avg. Time / Trial | Peak VRAM |
|-------|-------|-------------|-------------------|-----------|
| Baseline eval | TinyLlama-1.1B | 10 prompts | ~2 min | ~4 GB |
| SFT | TinyLlama-1.1B + LoRA | 9,000 samples | ~25–40 min | ~8–10 GB |
| DPO | SFT model + LoRA | 2,700 samples | ~20–30 min | ~10–12 GB |

---

## 8. Reproducibility

To reproduce all experiments:

### 8.1 Environment Setup

```bash
pip install transformers==4.44.2 peft==0.12.0 trl==0.9.6 \
    bitsandbytes==0.43.3 accelerate==0.34.2 datasets==2.21.0 \
    evaluate==0.4.3 bert-score==0.3.13 sacrebleu==2.4.3
```

### 8.2 Step-by-Step Execution

1. Open `00_baseline_evaluation.ipynb` on Colab (T4 GPU).
2. Fill in the `"reference"` fields in `test_prompts.json` using ChatGPT-4o.
3. Run all cells → outputs `baseline_results.csv` to Google Drive.
4. Open `01_sft_training.ipynb` on Colab (T4 GPU).
5. Run all cells (Cells 1–14) → outputs 5 adapter folders and `sft_results.csv` to Drive. Note the best adapter path printed in Cell 13.
6. Open `02_dpo_training.ipynb` on Colab (T4 GPU).
7. Update `BEST_SFT_ADAPTER_PATH`, `BEST_SFT_LORA_RANK`, `BEST_SFT_LORA_ALPHA`, and `BEST_SFT_LORA_MODULES` in Cell 4 and Cell 10 to match the best SFT trial.
8. Run all cells → outputs 5 DPO adapter folders and `dpo_results.csv` to Drive.

### 8.3 Prompt Template

All three notebooks use the Alpaca-style prompt template for generation:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
```

For training samples with a non-empty `input` field:

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

### 8.4 Random Seeds

All dataset shuffles and splits use `seed=42`. Model training uses the default PyTorch/Transformers random state (no explicit seed set for reproducibility of stochastic training; results may vary slightly across runs).

### 8.5 Best Model Parameters

*(Fill after completing experiments.)*

**Best SFT Trial (Trial ?)**

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | — |
| LoRA alpha | — |
| Target modules | — |
| Learning rate | — |
| Batch size | — |
| Gradient accumulation | — |
| Epochs | — |
| Val loss | — |
| Mean BLEU | — |
| Mean BERTScore F1 | — |

**Best DPO Trial (Trial ?)**

| Parameter | Value |
|-----------|-------|
| Beta | — |
| Learning rate | — |
| Batch size | — |
| Gradient accumulation | — |
| Epochs | — |
| Val loss | — |
| Mean BLEU | — |
| Mean BERTScore F1 | — |

---

## 9. References

1. Touvron, H., et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* arXiv:2302.13971.
2. Zhang, P., et al. (2024). *TinyLlama: An Open-Source Small Language Model.* arXiv:2401.02385.
3. Stanford Alpaca Team. (2023). *Stanford Alpaca: An Instruction-following LLaMA Model.* [GitHub](https://github.com/tatsu-lab/stanford_alpaca).
4. Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* arXiv:2305.18290.
5. Intel Labs. (2023). *Orca DPO Pairs.* HuggingFace Hub: `Intel/orca_dpo_pairs`.
6. Hu, E., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
7. Von Werra, L., et al. (2023). *TRL: Transformer Reinforcement Learning.* [GitHub](https://github.com/huggingface/trl).
8. Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314.
9. Zhang, T., et al. (2019). *BERTScore: Evaluating Text Generation with BERT.* arXiv:1904.09675.
10. Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores.* EMNLP 2018 (sacrebleu).
