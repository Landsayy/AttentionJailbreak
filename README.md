# AttentionJailbreak

### Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking

**ACL 2026**


---

## Overview

This repository contains the official implementation of **Attention-Guided Visual Jailbreaking** (Push-Pull Attack), which reveals a critical vulnerability in how Large Vision-Language Models (LVLMs) process safety instructions through their attention mechanism.

Our method achieves **94.4% attack success rate** on Qwen-VL (vs 68.8% baseline) by directly manipulating attention patterns rather than overpowering the model's safety alignment. The key insight is that LVLMs continuously retrieve safety instructions through attention, and this process can be circumvented through adversarial attention hijacking.

## Acknowledgements

This work builds upon and extends [Visual Adversarial Examples Jailbreak Large Language Models](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models) by [@Unispac](https://github.com/Unispac), which serves as our baseline. We thank the authors for their pioneering work on visual adversarial attacks against VLMs.

<p align="center"><img src="assets/method.png" width="85%"></p>


## Key Results

| Model | AdvBench | HarmBench | JailbreakBench | StrongREJECT |
|:------|:--------:|:---------:|:--------------:|:------------:|
| Qwen-VL-Chat | 94.4% | 95.5% | 90.4% | 92.0% |
| LLaVA-1.5-7B | 77.5% | 78.0% | 84.0% | 84.0% |
| InternVL2-8B | 18.3% | 17.5% | 19.0% | 15.3% |

> Attack Success Rate (ASR) measured by Llama Guard 3 safety classifier.

## Core Method: Push-Pull Attention Loss

The Push-Pull Attention Loss works by:

1. **SUPPRESS**: Reducing attention from generated tokens to system prompt tokens
2. **AMPLIFY**: Increasing attention from generated tokens to image tokens

$$\mathcal{L}_{\text{attn}} = \alpha \cdot \underbrace{\frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} \sum_{j \in \mathcal{S}} A_{ij}}_{\text{Push: target} \to \text{system}} - \beta \cdot \underbrace{\frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} \sum_{j \in \mathcal{V}} A_{ij}}_{\text{Pull: target} \to \text{image}}$$

where $\mathcal{T}$, $\mathcal{S}$, $\mathcal{V}$ denote target, system prompt, and image token sets, and $A_{ij}$ is the averaged attention weight over the last $K$ layers.

## Installation

```bash
git clone https://github.com/Landsayy/AttentionJailbreak.git
cd AttentionJailbreak
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch 2.1+
- CUDA 12.1+
- Transformers 4.45+

### Model Preparation

Download models from HuggingFace:

| Model | HuggingFace ID |
|:------|:--------------|
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` |
| Qwen-VL-Chat | `Qwen/Qwen-VL-Chat` |
| InternVL2-8B | `OpenGVLab/InternVL2-8B` |
| Llama Guard 3 | `meta-llama/Llama-Guard-3-8B` |

## Quick Start

### 1. Run Adversarial Attack

```bash
# LLaVA-1.5
python attack/attack.py \
    --model llava \
    --model_path /path/to/llava-1.5-7b-hf \
    --image_path images/clean.jpeg \
    --use_corpus \
    --num_iter 2000 \
    --eps 16 --alpha 1 \
    --constrained \
    --alpha_suppress 10.0 \
    --beta_amplify 5.0 \
    --save_dir ./results/llava_attack

# Qwen-VL
python attack/attack.py \
    --model qwen \
    --model_path /path/to/qwen-vl-chat \
    --image_path images/clean.jpeg \
    --use_corpus \
    --num_iter 2000 \
    --eps 16 --alpha 1 \
    --constrained \
    --alpha_suppress 10.0 \
    --beta_amplify 5.0 \
    --save_dir ./results/qwen_attack

# InternVL2
python attack/attack.py \
    --model internvl \
    --model_path /path/to/internvl2-8b \
    --image_path images/clean.jpeg \
    --use_corpus \
    --num_iter 2000 \
    --eps 16 --alpha 1 \
    --constrained \
    --alpha_suppress 10.0 \
    --beta_amplify 5.0 \
    --save_dir ./results/internvl_attack
```

### 2. Generate Model Responses

```bash
python evaluation/generate_responses.py \
    --model llava \
    --input_file harmful_corpus/input_advbench.json \
    --image_path results/llava_attack/adversarial.png \
    --output_file results/responses/llava_advbench.json \
    --model_path /path/to/llava-1.5-7b-hf
```

### 3. Evaluate Safety (ASR)

```bash
python evaluation/evaluate.py \
    --input_file results/responses/llava_advbench.json \
    --output_csv results/evaluation/llava_advbench_eval.csv \
    --benchmark advbench \
    --condition "PushPull" \
    --model_name "LLaVA-1.5"
```

### Or run the full pipeline:

```bash
bash scripts/run_pipeline.sh \
    --model llava \
    --model_path /path/to/llava-1.5-7b-hf \
    --benchmark advbench \
    --eps 16 \
    --num_iter 2000
```

## Project Structure

```
AttentionJailbreak/
├── attack/
│   └── attack.py              # Push-Pull adversarial attack implementation
├── evaluation/
│   ├── generate_responses.py  # Generate model responses
│   └── evaluate.py            # Safety evaluation (Llama Guard, Detoxify)
├── harmful_corpus/            # Benchmark datasets
│   ├── input_advbench.json
│   ├── input_harmbench.json
│   ├── input_jailbreakbench.json
│   ├── input_strongreject.json
│   └── derogatory_corpus.csv  # Target texts for optimization
├── images/
│   └── clean.jpeg             # Clean input image
├── scripts/
│   └── run_pipeline.sh       # End-to-end evaluation pipeline
├── requirements.txt
└── README.md
```

## Key Parameters

| Parameter | Default | Description |
|:----------|:-------:|:------------|
| `--eps` | 16 | Perturbation budget (in /255 units) |
| `--num_iter` | 2000 | Number of PGD iterations |
| `--alpha_suppress` | 10.0 | Weight for suppressing system attention |
| `--beta_amplify` | 5.0 | Weight for amplifying image attention |
| `--attn_layers` | last-6 | Target attention layers |

## Citation

```bibtex
@inproceedings{attentionjailbreak2026,
  title     = {Seeing No Evil: Blinding Large Vision-Language Models to Safety Instructions via Adversarial Attention Hijacking},
  author    = {Li, Jingru and Ren, Wei and Zhu, Tianqing},
  booktitle = {ACL 2026},
  year      = {2026}
}
```

## Disclaimer

This repository is released for **academic research purposes only**. The adversarial techniques demonstrated here are intended to advance understanding of VLM safety vulnerabilities and to motivate stronger defenses. Any misuse is strictly prohibited.
