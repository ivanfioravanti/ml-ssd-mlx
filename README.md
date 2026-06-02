# Simple Self-Distillation with MLX 
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2604.01193-b31b1b.svg)](https://arxiv.org/abs/2604.01193)
[![License](https://img.shields.io/badge/License-Apple-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)

### Embarrassingly Simple Self-Distillation Improves Code Generation

Ruixiang Zhang\*, Richard He Bai\*, Huangjie Zheng\*, Navdeep Jaitly, Ronan Collobert, Yizhe Zhang\*

<sub>\*Equal contribution</sub>

</div>

<p align="center">
  <img src="figures/fig_teaser.png" width="100%" alt="SSD Overview">
</p>

## ✨ Overview

This fork is an experimental Apple Silicon / MLX port of the original
[`apple/ml-ssd`](https://github.com/apple/ml-ssd) repository. The goal is to
reproduce the Simple Self-Distillation workflow locally with
[`mlx-lm`](https://github.com/ml-explore/mlx-lm) instead of vLLM.

This work is in an early validation phase. The data-generation and evaluation
paths have been migrated to MLX-LM and basic smoke tests pass, but benchmark
numbers should be treated as experimental until the generation, fine-tuning, and
evaluation setup has been validated end to end against the published settings.

The upstream repository reproduces the method from the paper:

> **Embarrassingly Simple Self-Distillation Improves Code Generation**

The approach consists of three simple steps:

1. **Sample** solutions from a frozen model at non-unit temperature  
2. **Fine-tune** on raw, unverified outputs using standard cross-entropy  
3. **Decode** with a separately tuned temperature  

**No rewards · No verifier · No teacher · No RL**

For full details, see the [paper](https://arxiv.org/abs/2604.01193).

---

## 📰 News

- **[2026-04-03]** 🚀 Initial release of repository  
- **[2026-04-03]** 🤗 Model checkpoints coming soon on Hugging Face
- **[2026-04-07]** 🤗 Model checkpoints released
- **[2026-04-16]** 🔧 Data generation pipeline released
- **[2026-05-31]** Experimental MLX-LM migration branch for local Apple Silicon runs
- *(More updates will be added here)*

---

## 🚀 Getting Started

```bash
git clone https://github.com/ivanfioravanti/ml-ssd-mlx.git
cd ml-ssd-mlx
uv sync --group evaluation          # for evaluation only
uv sync --group data-generation     # for data generation only
uv sync --group evaluation --group data-generation  # for both
```

This MLX version runs inference with `mlx-lm` on Apple Silicon. The upstream
`apple/ml-ssd` repository remains the source of truth for the original vLLM
workflow and published results.

<details>
<summary>Evaluation commands</summary>

```bash
source .venv/bin/activate
python evaluation/eval.py \
    --model mlx-community/Qwen3-4B-Instruct-2507-8bit \
    --max_tokens 65536 \
    --n_repeat 20 \
    --sampling_params "temperature=1.1,top_p=0.8,top_k=20,min_p=0.0" \
    --completion_batch_size 4 \
    --prefill_batch_size 2 \
    --output_path ./results/
```

For quick validation, add `--limit 1` or `--limit 5` and reduce
`--max_tokens`. Full benchmark-style runs should remove `--limit` and use the
sampling parameters from the model card.

</details>

<details>
<summary>Data generation</summary>

```bash
source .venv/bin/activate
python data_generation/generate.py --config data_generation/config.yaml
```

This runs the full pipeline end-to-end: loads the dataset, generates solutions with MLX-LM, and post-processes into chat-template JSONL for SFT training. Edit `data_generation/config.yaml` to change the model, dataset, sampling temperature, batch sizes, etc.

By default, `prompt_selection.mode: paper` selects **~10K unique prompts** from
`seed_sft` (dedupe by `question_id`, `filter_is_passed: true` — matching the
paper’s scale on the public Hugging Face split). Use `--no-paper-prompts` for
the full 591K-row split, or `--limit 20` for smoke tests on the paper set.

The current data-generation config follows the SimpleSD-4B-instruct
self-distillation settings: `temperature=1.6`, `top_p=0.8`, `top_k=20`. For
faster local experiments, an MLX-converted model such as
`mlx-community/Qwen3-4B-Instruct-2507-8bit` can be passed with `--model-name`.

</details>

<details>
<summary>Current experimental status</summary>

- Replaced vLLM dependencies with `mlx-lm`.
- Added an MLX text-generation adapter shared by evaluation and data generation.
- Kept `datasets==3.6.0` because LiveCodeBench v6 is still script-backed and
  cannot be loaded by `datasets` 4.x.
- Added `--limit` to evaluation for smoke tests and short validation runs.
- Fine-tuning is not fully integrated into this repo yet. Generated JSONL can be
  used with `mlx_lm lora` or another local training setup, but evaluating a
  locally trained SimpleSD adapter still needs adapter-loading support in the
  evaluation CLI.

</details>

## 🤗 Models
| Model | HuggingFace |
|:---|:---|
| SSD-4B-Instruct | [apple/SimpleSD-4B-instruct](https://huggingface.co/apple/SimpleSD-4B-instruct) |
| SSD-4B-Thinking | [apple/SimpleSD-4B-thinking](https://huggingface.co/apple/SimpleSD-4B-thinking) |
| SSD-30B-A3B-Instruct | [apple/SimpleSD-30b-a3b-instruct](https://huggingface.co/apple/SimpleSD-30b-instruct) |

## 📁 Repository Structure

```
├── data_generation/
│   ├── generate.py              # End-to-end data generation pipeline
│   ├── config.yaml              # Generation & post-processing config
│   └── templates/               # Prompt templates
├── evaluation/
│   ├── eval.py                  # CLI entry point
│   ├── benchmark.py             # LiveCodeBench v6 implementation
│   ├── mlx_generation.py        # MLX-LM generation adapter
│   └── livecodebench_utils.py   # Code execution utilities
├── figures/
│   └── fig_teaser.png
├── pyproject.toml
└── README.md
```

## 📝 Citation

```bibtex
@misc{zhang2026embarrassinglysimpleselfdistillationimproves,
      title={Embarrassingly Simple Self-Distillation Improves Code Generation},
      author={Ruixiang Zhang and Richard He Bai and Huangjie Zheng and Navdeep Jaitly and Ronan Collobert and Yizhe Zhang},
      year={2026},
      eprint={2604.01193},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.01193},
}
```
