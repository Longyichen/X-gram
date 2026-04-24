# X-gram

[English](./README.md) | [中文](./README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2604.21724-b31b1b.svg)](https://arxiv.org/abs/2604.21724)
[![License](https://img.shields.io/badge/license-Modified_MIT-blue.svg)](./LICENSE)

面向 Transformer 语言模型的高效 token-indexed memory injection 研究代码。

## About

**Description:** Plug-and-play research code for X-gram, an efficient
token-indexed memory injection framework for Transformer language models.

**Website:** https://arxiv.org/abs/2604.21724

**Topics:** `language-models`, `transformers`, `pytorch`, `olmo`,
`memory-augmentation`, `embedding-injection`, `token-indexed-memory`,
`shortconv`, `hashing`, `streaming-datasets`

X-gram 是一个基于 OLMo-core 的研究代码仓库，用于探索通过 token-indexed memory 扩展模型容量的计算解耦 scaling 路径。它在保持额外计算量和 activation 开销较小的前提下，引入频率感知的 hash routing、多尺度 ShortConv 特征抽取，以及可配置的 hidden state 和 attention pathway 注入机制。

核心思想是：与其让每个 token 都承担更大的 dense Transformer 计算，不如从轻量的 token-indexed memory 中检索特征，再用卷积式 extractor 将 1-gram lookup refinement 成局部 x-gram 信号，并注入到模型的指定路径中。这样可以获得一条 compute-decoupled capacity axis：它比僵硬的 2-gram/3-gram 查找表更自适应，也比朴素扩大查找表更适合 Zipf-like token 分布下的训练。

## 亮点

- **频率感知的 memory allocation。** 高频 VIP token 拥有独立容量，长尾 token 则被压缩到 probability-balanced hash buckets，并配合 alias rows 与 row-level gating。
- **X-gram 特征抽取。** 检索到的 1-gram 向量会经过轻量多尺度 ShortConv refinement，从静态 token lookup 转换为包含局部上下文的 x-gram 特征。
- **灵活的注入位置。** 支持配置 hidden residual path (`h`)、attention value stream (`v`)、query/key stream (`q`, `k`, `qk`) 以及 output projection (`o`)。
- **预算对齐的实验配置。** 仓库包含 X-gram 变体以及 Engram、Retoken、MoRT 等 baseline 的匹配配置，便于公平比较和 ablation。
- **兼容 OLMo 的训练栈。** 训练入口复用 OLMo-core 的配置、分布式训练、checkpoint 与 streaming data path。

## 为什么需要 X-gram？

普通 lookup-table augmentation 的吸引力在于：它能用很少的额外 FLOPs 增加大量静态参数。但实践中，单纯扩大查找表，或者显式枚举固定的 2-gram/3-gram entry，并不是优雅的 memory scaling 方式。它会把上下文粒度写死，长尾 token 对应的大量 row 难以被充分训练，并且在纵向扩展到更多 layer 或更多 view 时，平行 slot 很容易学到重复的 token identity，而不是新的有效信息。

X-gram 将 token-indexed memory 看作一个信息抽取问题，而不是更大的字典。它从高效的 1-gram retrieval 出发，再通过多尺度 ShortConv 对相邻检索特征进行科学化、适应性的多词元信息提取。相比固定 n-gram 扩展，这种方式更灵活：局部组合由卷积 refinement 学习得到，不同 kernel 捕获不同上下文跨度，模型可以在不显式物化所有 phrase table 的情况下得到有用的 x-gram 特征。

这个设计从本质上针对 lookup table 饱和的两个根因：

- **长尾可训练性。** Frequency-aware VIP+hash routing 将容量集中到更新密集的区域，并让长尾 token 共享压缩容量，避免大量直接 row 长期欠训练。
- **纵向扩展冗余。** View-specific ShortConv refinement 打破平行 lookup slot 的对称性，让新增 memory view 更容易形成不同的局部特征，而不是重复复制同一份 token identity。

## 方法概览

X-gram 将 lookup-based model augmentation 拆成三个阶段。

1. **Token-to-memory routing。** 将 token ID 映射到物理 memory row。X-gram 使用 probability-balanced hybrid hashing：高频 token 被放入保留的 VIP 区域，长尾 token 根据平滑后的经验频率分组，并映射到更紧凑的 bucket-local tables。这直接解决普通查找表中的长尾更新不均衡问题。
2. **Information extraction。** 每一路检索序列都会经过 view-specific ShortConv module。不同 kernel size 提供多尺度局部上下文，将 1-gram retrieval 转换为自适应 x-gram extraction，并帮助并行 memory views 避免 slot redundancy 与 representation collapse。
3. **Injection。** refined memory signal 通过 learned gates 融合，并注入到指定 Transformer pathway 中。推荐配置主要使用 attention value 与 hidden residual injection，因为它们在质量和预算之间有较好的 trade-off。

在配套论文设定中，X-gram 在 0.73B 与 1.15B 两个模型尺度下进行 matched backbone 和 matched training budget 的评测。目标是在使用更小、更易训练 lookup table 的同时，降低 validation perplexity 并提升 downstream accuracy。

## 设计原则

X-gram 的仓库设计目标不是保存一次性实验快照，而是尽可能即插即用，方便后续 lookup-memory 研究继续开发。

- **模块化 injection stack。** Routing、feature extraction 与 injection target 被拆分开，新的 memory mapping、ShortConv 变体或 target pathway 可以独立替换。
- **Config-first experiments。** 大部分结构选择都放在 YAML 中：injection targets、重复 views、hash maps、ShortConv kernels、warmup 行为等都可以不改训练代码直接调整。
- **兼容 OLMo-core。** 实现尽量贴近 OLMo-core 的 Transformer interface，方便复用现有分布式训练、checkpoint、logging 与标准 backbone 对比流程。
- **兼容 data-loader。** 数据转换围绕 `ubdataloader` 消费的 streaming shards 设计，整体流程与 Megatron-style pipeline 的实践保持一致：离线完成 tokenize/build/index/shard，训练阶段通过 indexed streaming reads 高吞吐读取。
- **面向研究扩展。** Baseline 与 ablation 配置和主方法放在同一套 launcher/data format/evaluation budget 下，方便后续工作只替换模块，不重写训练系统。

## 仓库结构

```text
.
|-- OLMo-core/                  # 带 injection 模块的 OLMo-core fork
|-- assets/
|   |-- frequency_stats/        # 用于 hash maps 的 token 频率统计
|   `-- token_maps/             # 预构建的 VIP+hash token maps
|-- configs/
|   |-- l10/                    # 10-layer 训练配置
|   `-- l12/                    # 12-layer 训练配置
|-- packages/
|   |-- olmo_in_loop_evals/     # 可选的 in-loop evaluation package
|   |-- streaming/              # 修改后的 streaming dataset package
|   `-- ubdataloader/           # 本地 data-loader package
|-- scripts/
|   `-- train/                  # 训练 launcher 与配置转换逻辑
|-- tools/                      # 工具脚本，包括 hash token-map builder
`-- pyproject.toml              # editable install 配置
```

仓库结构同样服务于“组件可替换”的设计目标：X-gram 相关模型逻辑集中在 `embedding_injection/`，训练 recipe 用独立 YAML 表达，数据读取单独封装为 package，utility scripts 不混入模型路径。因此这个仓库既能复现论文配置，也适合快速原型化新的 token-indexed memory 模块。

关键实现文件：

- `OLMo-core/src/olmo_core/nn/embedding_injection/xgram.py`：X-gram 模块构建与 runtime injection hooks。
- `OLMo-core/src/olmo_core/nn/embedding_injection/ops/hash_injection.py`：hash-token-map lookup module。
- `OLMo-core/src/olmo_core/nn/embedding_injection/ops/shortconv.py`：ShortConv refinement module。
- `scripts/train/olmo_train.py`：YAML/env parsing 与 OLMo training config assembly。
- `tools/build_hash_token_map.py`：frequency-aware VIP+hash map 生成工具。

## 安装

需要 Python 3.10 或更高版本。请从仓库根目录安装。

```bash
git clone git@github.com:Longyichen/X-gram.git
cd X-gram

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install -e ./packages/streaming
pip install -e ".[train]"
```

in-loop evaluation package 已经拆到单独仓库维护，因为它包含 tokenizer 文件、缓存的 Hugging Face dataset shards，以及预构建的 OpenEval requests。只有需要 eval-time workflow 时才需要安装：

```bash
git clone git@github.com:Longyichen/olmo_in_loop_evals.git packages/olmo_in_loop_evals
pip install -e ./packages/olmo_in_loop_evals
```

也可以不放在 `packages/` 下，直接从 GitHub 安装：

```bash
pip install git+ssh://git@github.com/Longyichen/olmo_in_loop_evals.git
```

X-gram 主仓库会继续 ignore `packages/olmo_in_loop_evals/`，保持主代码仓库轻量。

## 快速开始

启动默认 `l10` X-gram 配置：

```bash
bash ./scripts/train/olmo_train.sh ./configs/l10/xgram_3h_2v_hash.yaml
```

launcher 会读取 YAML，配置 OLMo-core，设置本地 `PYTHONPATH`，准备 streaming cache path，并启动分布式训练任务。常见分布式环境变量如下：

```bash
GPUS_PER_NODE=8 \
NNODES=1 \
MASTER_ADDR=localhost \
MASTER_PORT=6000 \
bash ./scripts/train/olmo_train.sh ./configs/l10/xgram_3h_2v_hash.yaml
```

YAML 配置控制模型规模、数据路径、batch size、训练 token 数量以及 injection recipe。实际运行时通常需要先修改 `data` 部分，让它指向自己的 streaming dataset 与 tokenizer 路径。

## 配置

`configs/` 目录包含 X-gram recipes 和 baseline recipes，并按模型规模组织在 `configs/l10/` 与 `configs/l12/` 下。

| 配置 | 用途 |
| --- | --- |
| `xgram_3h_2v_hash.yaml` | 推荐 X-gram recipe，包含 hidden/value injection 与 hash routing。 |
| `xgram_2v_hash.yaml` | 双 view value-stream 的 X-gram hash routing 配置。 |
| `qkvoh-hash-share.yaml` | Q/K/V/O/H 全路径注入，包含 shared QK path 与 hash routing。 |
| `qk-share.yaml` | 不使用 hash routing 的 shared QK injection。 |
| `1h1v1o.yaml` | 较小的 non-hashed X-gram ablation。 |
| `engram.yaml` | [Engram paper](https://arxiv.org/abs/2601.07372) baseline。 |
| `retoken.yaml` | [ReToken paper](https://openreview.net/forum?id=VjAOHI1owB) baseline。 |
| `mort.yaml` | [MoRT paper](https://openreview.net/forum?id=VjAOHI1owB) baseline。 |

一个典型 X-gram config 的 `embedding_injection` 配置如下：

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

Layer list 中可以重复同一个 layer index。重复代表在同一个 injection site 上创建多个独立 view，也就是配置中表达 `1x`、`2x`、`4x` memory-capacity setting 的方式。

## Token Maps

默认 hashed configs 使用的 token map 已包含在仓库中：

```text
assets/token_maps/injection_token_map_alpha0_5_M32_Cap75968_mc3_mwd0_8.npz
```

它由下面的频率统计文件生成：

```text
assets/frequency_stats/qwen_streaming_freqs_41B.json
```

生成新 map：

```bash
python3 ./tools/build_hash_token_map.py \
  --freq-path ./assets/frequency_stats/qwen_streaming_freqs_41B.json
```

常用参数示例：

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

## 数据处理

训练数据管线基于 [Databricks AI Research](https://www.databricks.com/research/databricks-ai-research) 数据，从 [DCLM baseline 1.0 dataset](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) 开始处理。预期流程如下：

1. 从 Hugging Face 下载或流式读取 DCLM baseline 1.0。
2. 使用目标 tokenizer 对语料进行 tokenize。
3. 将 tokenized examples 转换成兼容 `packages/streaming/` 的 streaming shards。
4. 训练时通过 `ubdataloader` 读取这些 shards。

这与 Megatron-style preprocessing 的设计理念一致：昂贵的数据清洗、tokenization、indexing 和 sharding 在离线阶段完成；训练阶段只读取紧凑的 indexed representation，并获得确定性 chunking 与高吞吐读取。在 X-gram 中，这套格式适配到本地 `ubdataloader`，因此实验可以复用相同的数据接口，只改变模型侧 injection modules。

## 数据路径要求

训练使用 `packages/streaming/` 下的修改版 streaming data pipeline，以及 `packages/ubdataloader/` 下的本地 loader。默认示例配置包含原实验环境中的 `/tmp/...` placeholder path。大规模运行前需要替换：

- `data.streaming_data_path`
- `data.streaming_tokenizer_model`
- `data.streaming_ckpt_path`
- `run.save_root`

launcher 也支持通过环境变量覆盖路径，例如 `STREAMING_DATA_PATH`、`STREAMING_CACHE_BASE` 和 `RUN_NAME`。

## Weights & Biases

训练脚本会读取并导出已有的 W&B 环境变量：

```bash
export WANDB_API_KEY=...
export WANDB_PROJECT=xgram
export WANDB_ENTITY=...
export WANDB_MODE=online
```

本地 dry run 可设置 `WANDB_MODE=offline` 或 `WANDB_MODE=disabled`。

## 开发

该仓库是研究工作区的 curated open-source copy。根 package 会从 `OLMo-core/src` 安装 `olmo_core`，并从 `packages/ubdataloader/src` 安装 `ubdataloader`。

```bash
pip install -e ./packages/streaming
pip install -e ".[train]"
```

如果要添加新的 injection mode，通常需要修改：

1. 在 `OLMo-core/src/olmo_core/nn/embedding_injection/` 下新增或更新模块。
2. 如果走 registry 路径，在 `registry.py` 中注册 mode metadata。
3. 在 `scripts/train/olmo_train.py` 中扩展 YAML/env parsing。
4. 在 `configs/` 下添加一个能覆盖新 mode 的小配置。

## 引用

如果本仓库对你的研究有帮助，请引用下面这篇论文：

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

## 协议

本项目使用 Modified MIT License。该协议保留 MIT 的许可授权，并增加商业使用署名要求：如果该软件或衍生作品被用于商业产品或服务，应在对应产品或服务的用户界面中显著展示 “X-gram”。完整文本见 [LICENSE](./LICENSE)。

## 致谢

X-gram 基于 OLMo-core 训练栈构建，并包含研究实验中使用的 streaming-data 与 data-loader 工具的本地修改版本。
