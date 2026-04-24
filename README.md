# X-gram

[English](./README.md) | [中文](./README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2604.21724-b31b1b.svg)](https://arxiv.org/abs/2604.21724)
[![License](https://img.shields.io/badge/license-Modified_MIT-blue.svg)](./LICENSE)

Efficient token-indexed memory injection for Transformer language models.

## About

**Description:** Plug-and-play research code for X-gram, an efficient
token-indexed memory injection framework for Transformer language models.

**Website:** https://arxiv.org/abs/2604.21724

**Topics:** `language-models`, `transformers`, `pytorch`, `olmo`,
`memory-augmentation`, `embedding-injection`, `token-indexed-memory`,
`shortconv`, `hashing`, `streaming-datasets`

X-gram is the research codebase for a lookup-based scaling method that expands
model capacity through token-indexed memory while keeping the extra compute and
activation cost small. It builds on an OLMo-core fork and adds frequency-aware
hash routing, multi-scale ShortConv feature extraction, and flexible injection
targets for hidden states and attention pathways.

The central idea is simple: instead of increasing dense Transformer compute for
every token, X-gram retrieves lightweight token-indexed features, refines them
into local x-gram signals with convolutional extractors, and injects them into
selected parts of the model. This gives the model a compute-decoupled capacity
axis that is more adaptive than rigid 2-gram/3-gram lookup tables and more
trainable under Zipf-like token statistics than naive table scaling.

## Highlights

- **Frequency-aware memory allocation.** VIP tokens receive dedicated capacity,
  while the long tail is compressed into probability-balanced hash buckets with
  alias rows and row-level gating.
- **X-gram feature extraction.** Retrieved 1-gram vectors are refined with
  lightweight multi-scale ShortConv modules, turning static token lookup into
  local context features.
- **Flexible injection sites.** Configure injection into hidden residual paths
  (`h`), attention value streams (`v`), query/key streams (`q`, `k`, `qk`), or
  output projections (`o`).
- **Budget-aware experiments.** The repository includes matched configurations
  for X-gram variants and baselines such as Engram, Retoken, and MoRT.
- **OLMo-compatible training stack.** Training entrypoints reuse the OLMo-core
  configuration, distributed training, checkpointing, and streaming data path.

## Why X-gram?

Plain lookup-table augmentation is attractive because it can add many static
parameters with little extra FLOPs. In practice, however, simply enlarging a
table or enumerating fixed 2-gram/3-gram entries is a brittle way to scale
memory. It hard-codes the granularity of context, wastes capacity on sparse
long-tail rows, and tends to create redundant parallel slots when scaled
vertically across layers or views.

X-gram treats token-indexed memory as an information-extraction problem rather
than a larger dictionary. It starts from efficient 1-gram retrieval, then uses
multi-scale ShortConv modules to extract principled, adaptive multi-token
signals from nearby retrieved features. Compared with fixed n-gram expansion,
this is less rigid: local composition is learned through convolutional
refinement, different kernels capture different context spans, and the model can
derive useful x-gram features without explicitly materializing every possible
phrase table.

This design targets two root causes behind lookup-table saturation:

- **Long-tail trainability.** Frequency-aware VIP+hash routing concentrates
  capacity where updates are dense and shares tail capacity where direct rows
  would remain under-trained.
- **Vertical scaling redundancy.** View-specific ShortConv refinement breaks the
  symmetry of parallel lookup slots, so additional memory views can become
  diverse local features instead of repeated copies of the same token identity.

## Method Overview

X-gram factors lookup-based model augmentation into three stages.

1. **Token-to-memory routing.** Token IDs are mapped to physical memory rows.
   X-gram uses probability-balanced hybrid hashing: frequent tokens are placed
   in a reserved VIP region, while tail tokens are grouped by smoothed empirical
   frequency and mapped into compact bucket-local tables. This directly attacks
   the long-tail update imbalance of ordinary lookup tables.
2. **Information extraction.** Each retrieved sequence is passed through a
   view-specific ShortConv module. Different kernel sizes provide multi-scale
   local context, turning 1-gram retrieval into adaptive x-gram extraction and
   helping parallel memory views avoid redundant slot collapse.
3. **Injection.** The refined memory signal is fused with learned gates and
   injected into chosen Transformer pathways. The recommended configurations
   emphasize attention value and hidden residual injection because they provide a
   strong quality-to-budget trade-off.

In the accompanying paper draft, X-gram is evaluated under matched backbone and
training budgets at 0.73B and 1.15B scales. The method is designed to reduce
validation perplexity and improve downstream accuracy while using smaller and
more trainable lookup tables than naive token-indexed scaling.

## Design Principles

X-gram is organized to be plug-and-play for future lookup-memory research rather
than a one-off experiment snapshot.

- **Modular injection stack.** Routing, feature extraction, and injection targets
  are separated so new memory mappings, ShortConv variants, or target pathways
  can be developed independently.
- **Config-first experiments.** Most architectural choices live in YAML:
  injection targets, repeated views, hash maps, ShortConv kernels, and warmup
  behavior can be changed without editing training code.
- **OLMo-core compatibility.** The implementation stays close to the OLMo-core
  Transformer interface, making it easier to compare against standard backbones
  and reuse existing distributed training, checkpointing, and logging paths.
- **Data-loader compatibility.** Dataset conversion is designed around streaming
  shards consumed by `ubdataloader`, with a processing flow that mirrors the
  practical separation used by Megatron-style pipelines: offline tokenize/build
  once, then train through indexed streaming reads.
- **Research extensibility.** Baselines and ablations live beside the main
  method, so follow-up work can swap modules while keeping the same launcher,
  data format, and evaluation budget.

## Repository Layout

```text
.
|-- OLMo-core/                  # Vendored OLMo-core fork with injection modules
|-- assets/
|   |-- frequency_stats/        # Token-frequency statistics for hash maps
|   `-- token_maps/             # Prebuilt VIP+hash token maps
|-- configs/
|   |-- l10/                    # 10-layer training configs
|   `-- l12/                    # 12-layer training configs
|-- packages/
|   |-- olmo_in_loop_evals/     # Optional in-loop evaluation package
|   |-- streaming/              # Modified streaming dataset package
|   `-- ubdataloader/           # Local data-loader package
|-- scripts/
|   `-- train/                  # Training launcher and config translation
|-- tools/                      # Utilities, including hash token-map builder
`-- pyproject.toml              # Editable install definition
```

The repository layout follows the same design goal as the method itself:
components should be easy to replace. X-gram-specific model logic is isolated
under `embedding_injection/`, training recipes are expressed as standalone YAML
files, data readers are packaged separately, and utility scripts are kept out of
the model path. This makes the repository suitable for both reproducing the
paper recipes and quickly prototyping new token-indexed memory modules.

Important implementation files:

- `OLMo-core/src/olmo_core/nn/embedding_injection/xgram.py`: X-gram module
  construction and runtime injection hooks.
- `OLMo-core/src/olmo_core/nn/embedding_injection/ops/hash_injection.py`:
  hash-token-map lookup module.
- `OLMo-core/src/olmo_core/nn/embedding_injection/ops/shortconv.py`:
  ShortConv refinement module.
- `scripts/train/olmo_train.py`: YAML/env parsing and OLMo training config
  assembly.
- `tools/build_hash_token_map.py`: frequency-aware VIP+hash map generation.

## Installation

Use Python 3.10 or newer. Install from the repository root.

```bash
git clone git@github.com:Longyichen/X-gram.git
cd X-gram

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install -e ./packages/streaming
pip install -e ".[train]"
```

The in-loop evaluation package is maintained as a separate repository because it
contains packaged tokenizer files, cached Hugging Face dataset shards, and
prebuilt OpenEval requests. Install it only when you need eval-time workflows:

```bash
git clone git@github.com:Longyichen/olmo_in_loop_evals.git packages/olmo_in_loop_evals
pip install -e ./packages/olmo_in_loop_evals
```

You can also install it directly without placing it under `packages/`:

```bash
pip install git+ssh://git@github.com/Longyichen/olmo_in_loop_evals.git
```

The X-gram repository keeps `packages/olmo_in_loop_evals/` ignored so the main
code checkout stays lightweight.

## Quick Start

Launch the default `l10` X-gram configuration:

```bash
bash ./scripts/train/olmo_train.sh ./configs/l10/xgram_3h_2v_hash.yaml
```

The launcher reads the YAML file, configures OLMo-core, sets the local
`PYTHONPATH`, prepares streaming cache paths, and starts the distributed training
job. Common distributed environment variables are supported:

```bash
GPUS_PER_NODE=8 \
NNODES=1 \
MASTER_ADDR=localhost \
MASTER_PORT=6000 \
bash ./scripts/train/olmo_train.sh ./configs/l10/xgram_3h_2v_hash.yaml
```

The YAML config controls model size, data paths, batch sizes, training tokens,
and the injection recipe. You will usually need to edit the `data` section to
point to your own streaming dataset and tokenizer paths.

## Configurations

The `configs/` directory contains both X-gram recipes and baseline recipes,
organized by model scale under `configs/l10/` and `configs/l12/`.

| Config | Purpose |
| --- | --- |
| `xgram_3h_2v_hash.yaml` | Recommended X-gram recipe with hidden and value injection plus hash routing. |
| `xgram_2v_hash.yaml` | Two-view value-stream X-gram with hash routing. |
| `qkvoh-hash-share.yaml` | Broad Q/K/V/O/H injection with shared QK path and hash routing. |
| `qk-share.yaml` | QK shared injection without hash routing. |
| `1h1v1o.yaml` | Smaller non-hashed X-gram ablation. |
| `engram.yaml` | [Engram paper](https://arxiv.org/abs/2601.07372) baseline. |
| `retoken.yaml` | [ReToken paper](https://openreview.net/forum?id=VjAOHI1owB) baseline. |
| `mort.yaml` | [MoRT paper](https://openreview.net/forum?id=VjAOHI1owB) baseline. |

An X-gram config includes an `embedding_injection` block like:

```yaml
embedding_injection:
  mode: X-gram
  targets: [h, v]
  h_layers: [0, 0, 0, 1, 1, 1]
  v_layers: [0, 0, 1, 1]
  shortconv_enabled: true
  shortconv_kernels: [3, 5, 7]
  hash_enabled: true
  hash_token_map_path: ./assets/token_maps/injection_token_map_alpha0_5_M32_Cap75968_mc3_mwd0_8.npz
  lambda_warmup_enabled: true
```

Layer lists may repeat indices. Repetition creates multiple independent views at
the same injection site, which is how the configs express `1x`, `2x`, and `4x`
style memory-capacity settings.

## Token Maps

The checked-in token map used by the default hashed configs is:

```text
assets/token_maps/injection_token_map_alpha0_5_M32_Cap75968_mc3_mwd0_8.npz
```

It is generated from:

```text
assets/frequency_stats/qwen_streaming_freqs_41B.json
```

To build a new map:

```bash
python3 ./tools/build_hash_token_map.py \
  --freq-path ./assets/frequency_stats/qwen_streaming_freqs_41B.json
```

Useful options include:

```bash
python3 ./tools/build_hash_token_map.py \
  --freq-path ./assets/frequency_stats/qwen_streaming_freqs_41B.json \
  --bucket-count 32 \
  --alpha 0.5 \
  --top-k 200 \
  --total-capacity 75968 \
  --max-copies 3 \
  --multi-weight-decay 0.8
```

## Data Processing

The training data pipeline is built around
[Databricks AI Research](https://www.databricks.com/research/databricks-ai-research)
data, starting from the
[DCLM baseline 1.0 dataset](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0).
The expected workflow is:

1. Download or stream DCLM baseline 1.0 from Hugging Face.
2. Tokenize the corpus with the target tokenizer.
3. Convert tokenized examples into streaming shards compatible with
   `packages/streaming/`.
4. Read those shards through `ubdataloader` during training.

This mirrors the design philosophy of Megatron-style preprocessing: expensive
data normalization, tokenization, indexing, and sharding are performed offline;
training then consumes a compact indexed representation with deterministic
chunking and high-throughput reads. In X-gram, that format is adapted to the
local `ubdataloader` path so experiments can reuse the same data interface while
changing only model-side injection modules.

## Data Expectations

Training uses the modified streaming data pipeline under `packages/streaming/`
and the local loader package under `packages/ubdataloader/`. The default example
configs contain placeholder `/tmp/...` paths from the original experiment
environment. Replace these fields before running at scale:

- `data.streaming_data_path`
- `data.streaming_tokenizer_model`
- `data.streaming_ckpt_path`
- `run.save_root`

The launcher also supports overriding paths through environment variables such
as `STREAMING_DATA_PATH`, `STREAMING_CACHE_BASE`, and `RUN_NAME`.

## Weights & Biases

The training script exports W&B environment variables if they are present:

```bash
export WANDB_API_KEY=...
export WANDB_PROJECT=xgram
export WANDB_ENTITY=...
export WANDB_MODE=online
```

Set `WANDB_MODE=offline` or `WANDB_MODE=disabled` for local dry runs.

## Development

This repository is a curated open-source copy of the research workspace. The
root package installs `olmo_core` from `OLMo-core/src` and `ubdataloader` from
`packages/ubdataloader/src`.

```bash
pip install -e ./packages/streaming
pip install -e ".[train]"
```

If you add a new injection mode, the usual touch points are:

1. Add or update modules under `OLMo-core/src/olmo_core/nn/embedding_injection/`.
2. Register mode metadata in `registry.py` if it follows the registry path.
3. Extend YAML/env parsing in `scripts/train/olmo_train.py`.
4. Add a small config under `configs/` that exercises the new mode.

## Citation

If this repository is useful for your work, please cite the paper:

```bibtex
@misc{chen2026ngramdataawarexgramextraction,
  title={Beyond N-gram: Data-Aware X-GRAM Extraction for Efficient Embedding Parameter Scaling},
  author={Yilong Chen and Yanxi Xie and Zitian Gao and He Xin and Yihao Xiao and Renbiao Liu and Haoming Luo and Yifan Luo and Zhengmao Ye and Tingwen Liu and Xin Zhao and Ran Tao and Bryan Dai},
  year={2026},
  eprint={2604.21724},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2604.21724},
}
```

## License

This project is released under a Modified MIT License. The license follows the
MIT permission grant and adds a commercial-use attribution requirement: if the
software or derivative works are used in a commercial product or service, the
product or service should prominently display "X-gram" in its user interface.
See [LICENSE](./LICENSE) for the full text.

## Acknowledgements

X-gram builds on the OLMo-core training stack and includes modified local copies
of streaming-data and data-loader utilities used by the research experiments.
