<div align="center">

# 🧠 Inference Collective

**Open-Source AI Tooling, Reproducible Model Benchmarks, 25 Hands-on Labs & AI Safety Intelligence**

[![Deploy GitHub Pages](https://img.shields.io/github/actions/workflow/status/inferencecollective/inferencecollective.github.io/pages.yml?branch=main&label=Deploy%20Pages&style=for-the-badge&logo=githubactions&logoColor=white&color=059669)](https://inferencecollective.github.io/)
[![Hosted AI Tools](https://img.shields.io/badge/AI_Tools-1%2C000%2B_Curated-8b5cf6?style=for-the-badge&logo=anthropic&logoColor=white)](https://inferencecollective.github.io/tools)
[![Open Weight Models](https://img.shields.io/badge/Open_Models-22%2B_Benchmarks-0ea5e9?style=for-the-badge&logo=huggingface&logoColor=white)](https://inferencecollective.github.io/models)
[![Interactive Labs](https://img.shields.io/badge/Interactive_Labs-25_Guides-10b981?style=for-the-badge&logo=python&logoColor=white)](https://inferencecollective.github.io/labs)
[![Built with Elixir](https://img.shields.io/badge/Built_With-Elixir_%7C_Phoenix-4B275F?style=for-the-badge&logo=elixir&logoColor=white)](https://elixir-lang.org/)
[![Tested with PyTorch](https://img.shields.io/badge/Tested-Python_3.13_%7C_PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-3b82f6?style=for-the-badge)](LICENSE)

<br />

[**🌐 Live Website**](https://inferencecollective.github.io/) &nbsp;•&nbsp;
[**🧰 Tools & Ecosystem**](https://inferencecollective.github.io/tools) &nbsp;•&nbsp;
[**🤖 Models & VRAM Leaderboard**](https://inferencecollective.github.io/models) &nbsp;•&nbsp;
[**🧪 25 Interactive Labs**](https://inferencecollective.github.io/labs) &nbsp;•&nbsp;
[**📝 Field Notes & Blog**](https://inferencecollective.github.io/blog) &nbsp;•&nbsp;
[**🗺️ 1,000-Item Roadmap**](https://inferencecollective.github.io/roadmap) &nbsp;•&nbsp;
[**🛡️ AI Security Matrix**](https://inferencecollective.github.io/security) &nbsp;•&nbsp;
[**✨ Contribute**](#-contributing-to-inference-collective)

---
</div>

## Guided ML engineering workshops

Start at `/workshops` for four tracks: **ML foundations**, **PyTorch**, **image preprocessing**, and **MLX on Apple silicon**. The first three are CPU friendly; MLX uses Metal GPU or CPU on a compatible Mac. Each pairs runnable source with expected outputs and an extension exercise. No API key is required. Use Python 3.11–3.13 and run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/workshops/ml-foundations.txt
python -m labs.workshop ml-foundations --check
python -m labs.workshop ml-foundations --step 1
python -m labs.workshop ml-foundations
```

For Windows PowerShell, create the environment with `py -3.12 -m venv .venv` and use `.venv\Scripts\python.exe` instead of `python`. Change `ml-foundations` to `pytorch` or `vision` in both the requirements path and runner command to choose another track. Requirements use bounded compatible versions; they are not an exact environment lock.

For Apple silicon Macs running macOS 14+ with native arm64 Python, install `requirements/workshops/mlx.txt` and run `python -m labs.workshop mlx --check`, then `python -m labs.workshop mlx`. Use `--device cpu` or `--device gpu` to select a backend explicitly. The MLX labs save a regression report and a warmed, synchronized compilation benchmark as JSON; they do not assume compilation is always faster. See `/labs/mlx-apple-silicon` for explanations and official documentation.

The website and runner share `priv/static/aidatafoundation/data/workshops.json`. To contribute a track, add its requirements and catalog entry with existing source modules, guide IDs, expected outcomes, and extension prompts. Run `python -m pytest tests/test_workshops.py` and `mix precommit` before submitting. For a live workshop, reserve 10 minutes for setup, pair learners for each exercise, and close with a demo of an output and a parameter change. Checkboxes record self-reported progress in browser storage; they do not verify execution.

## 📌 What You'll Find

| Section | Focus & Capabilities | Direct Link |
| :--- | :--- | :---: |
| 🧰 **Tools & Ecosystem Directory** | Curated registry of 70+ hosted LF AI & Data Foundation projects (Milvus, ONNX, Docling, Flyte, Feast) alongside Model Context Protocol (MCP) servers, Vector DBs, and runtime tools. | [Explore Tools ↗](https://inferencecollective.github.io/tools) |
| 🤖 **Open Models & VRAM Calculator** | Real-time leaderboard of 22+ frontier open models (DeepSeek V3, Qwen 2.5, LLaMA 3.3, Gemma 2, Mistral) with benchmark ratings and interactive GPU memory calculators. | [Calculate VRAM ↗](https://inferencecollective.github.io/models) |
| 🧪 **25 Interactive Labs & Runbooks** | Hands-on tutorials with runnable Python 3.13 code covering PyTorch, TorchVision, TensorFlow, Keras 3, SciPy, Pandas, Scikit-learn, XGBoost, Computer Vision, and Weaviate. | [Browse Labs ↗](https://inferencecollective.github.io/labs) |
| 🛡️ **AI Security & Threat Intelligence** | Threat vectors, guardrails, prompt injection scanners (Garak, Promptfoo, DeepEval), model watermarking, and agent sandboxing frameworks. | [Inspect Security ↗](https://inferencecollective.github.io/security) |
| 📝 **Engineering Blog & Field Notes** | Deep technical breakdowns on Qwen Coder models, Model Context Protocol (MCP) impact on SDLC, reproducible benchmarking, and AI data engineering. | [Read Field Notes ↗](https://inferencecollective.github.io/blog) |
| 🗺️ **1,000-Item Content Roadmap** | Four structured learning tracks across LLM architectures, agentic protocols, RAG, and AI safety with scheduled release milestones. | [View Roadmap ↗](https://inferencecollective.github.io/roadmap) |

---

## 🤖 Open Weight Model Leaderboard & VRAM Calculator

> ⚡ **Interactive Memory Physics**: Estimate exact GPU requirements for FP16, 8-bit, and 4-bit quantizations with dynamic context length scaling ($VRAM = \text{Weights} + \text{KV Cache} + \text{Activation Overhead}$).

### Verified Frontier Open Models

| Model | Maintainer | Parameters | Context | Benchmark | Min VRAM | License |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **[Qwen 2.5 32B](https://inferencecollective.github.io/models)** | Alibaba Cloud / Qwen | `32B` | 128K | **76.0%** | `65.8 GB` | `Apache 2.0` |
| **[Qwen 2.5 Coder 32B Instruct](https://inferencecollective.github.io/models)** | Alibaba Cloud / Qwen | `32B` | 32K | **68.0%** | `65.8 GB` | `Apache 2.0` |
| **[DeepSeek V3](https://inferencecollective.github.io/models)** | DeepSeek AI | `685B (37B active)` | 160K | **67.0%** | `171.8 GB` | `DeepSeek Open` |
| **[Phi 4](https://inferencecollective.github.io/models)** | Microsoft | `14B` | 16K | **67.0%** | `29.4 GB` | `MIT` |
| **[DeepSeek R1](https://inferencecollective.github.io/models)** | DeepSeek AI | `685B (37B active)` | 160K | **61.0%** | `180.4 GB` | `MIT` |
| **[Llama 3.1 Nemotron 49B](https://inferencecollective.github.io/models)** | NVIDIA | `49B` | 128K | **60.0%** | `100.2 GB` | `NVIDIA Open` |
| **[Mistral Small 24B Instruct](https://inferencecollective.github.io/models)** | Mistral AI | `24B` | 32K | **60.0%** | `47.3 GB` | `Apache 2.0` |
| **[Gemma 2 27B](https://inferencecollective.github.io/models)** | Google DeepMind | `27B` | 8K | **59.0%** | `54.2 GB` | `Gemma License` |

### 🧮 VRAM Formula & Estimation Model
$$\text{Total VRAM} = \frac{\text{Params} \times \text{Bytes per Weight}}{10^9} + \frac{2 \times \text{Layers} \times \text{Heads} \times \text{HeadDim} \times \text{Tokens} \times \text{Batch} \times \text{Bytes per Activation}}{10^9} + \text{Overhead}$$

- **16-bit Full Precision (FP16/BF16)**: $\approx 2.0 \text{ GB}$ per billion parameters.
- **8-bit Quantization (INT8/FP8)**: $\approx 1.0 \text{ GB}$ per billion parameters.
- **4-bit Quantization (INT4/AWQ/GGUF)**: $\approx 0.55 \text{ GB}$ per billion parameters.

---

## 🧪 25 Interactive Hands-on Labs & Runbooks

All 25 learning guides are self-contained and run on standard CPU environments with original, reproducible synthetic datasets and verified test assertions.

| # | Learning Path / Module | Core Focus & Technologies | Level | Lab Guide |
| :-: | :--- | :--- | :-: | :---: |
| 01 | **LangChain Practical Labs** | Agents, prompt chains, memory, tool dispatching & RAG | `Intermediate` | [Start Lab ↗](https://inferencecollective.github.io/labs/LangChain-Practical-Labs) |
| 02 | **AI-Native Vector Database** | Weaviate, Ollama, semantic search & HNSW indexing | `Intermediate` | [Start Lab ↗](https://inferencecollective.github.io/labs/AI-native-database) |
| 03 | **PyTorch Complete Guide** | Tensors, autograd, neural networks & transformer basics | `Foundational` | [Start Lab ↗](https://inferencecollective.github.io/labs/pytorch-complete-guide) |
| 04 | **TorchVision Complete Guide** | Vision datasets, geometry-aware transforms & CNN classifiers | `Hands-on` | [Start Lab ↗](https://inferencecollective.github.io/labs/torchvision-complete-guide) |
| 05 | **TensorFlow Complete Guide** | tf.data, custom Keras layers, SavedModel & LiteRT export | `Intermediate` | [Start Lab ↗](https://inferencecollective.github.io/labs/tensorflow-complete-guide) |
| 06 | **Keras 3 Multi-Backend Guide** | Portable neural nets targeting PyTorch, JAX, and TensorFlow | `Foundational` | [Start Lab ↗](https://inferencecollective.github.io/labs/keras-complete-guide) |
| 07 | **Data Quality Metrics for LLMs** | Token perplexity, deduplication, toxicity & bias scoring | `Intermediate` | [Start Lab ↗](https://inferencecollective.github.io/labs/data-quality-metrics) |
| 08 | **Ethical Synthetic Data Generation** | Differential privacy, statistical fidelity & GAN safety | `Advanced` | [Start Lab ↗](https://inferencecollective.github.io/labs/ethical-synthetic-data) |
| 09 | **LLM Evaluation Framework** | ROUGE, BLEU, BERTScore & automated LLM-as-a-judge | `Intermediate` | [Start Lab ↗](https://inferencecollective.github.io/labs/llm-evaluation) |
| 10 | **Reinforcement Learning Lab** | Q-learning, policy gradients, Markov decision processes | `Intermediate` | [Start Lab ↗](https://inferencecollective.github.io/labs/reinforcement-learning) |
| 11 | **Complete NumPy Tutorial** | Vectorization, broadcasting, strides & linear algebra | `Foundational` | [Start Lab ↗](https://inferencecollective.github.io/labs/numpy-in-depth) |
| 12 | **SciPy Scientific Computing** | Optimization, numerical integration & signal filtering | `Hands-on` | [Start Lab ↗](https://inferencecollective.github.io/labs/scipy-complete-guide) |
| 13 | **Pandas Data Analytics** | High-performance DataFrames, groupby, joins & timeseries | `Foundational` | [Start Lab ↗](https://inferencecollective.github.io/labs/pandas-complete-guide) |
| 14 | **Matplotlib Complete Guide** | Figures, axes, custom colormaps & publication charts | `Hands-on` | [Start Lab ↗](https://inferencecollective.github.io/labs/matplotlib-complete-guide) |
| 15 | **Seaborn Statistical Viz** | Distributions, pairplots, facets & categorical estimation | `Hands-on` | [Start Lab ↗](https://inferencecollective.github.io/labs/seaborn-complete-guide) |
| 16 | **Scikit-learn Complete Guide** | Classifiers, regressors, pipelines & cross-validation | `Foundational` | [Start Lab ↗](https://inferencecollective.github.io/labs/scikit-learn-complete-guide) |
| 17 | **XGBoost Complete Guide** | Gradient boosted trees, DMatrix, objective functions & pruning | `Intermediate` | [Start Lab ↗](https://inferencecollective.github.io/labs/xgboost-complete-guide) |
| 18 | **LightGBM Learning Guide** | Histogram-based boosting, leaf-wise tree growth & speed | `Intermediate` | [Start Lab ↗](https://inferencecollective.github.io/labs/lightgbm-complete-guide) |
| 19 | **CatBoost Complete Guide** | Categorical feature handling, Pool objects & GPU training | `Intermediate` | [Start Lab ↗](https://inferencecollective.github.io/labs/catboost-complete-guide) |
| 20 | **Pillow Image Processing** | RGBA buffers, spatial filters, transformations & batching | `Hands-on` | [Start Lab ↗](https://inferencecollective.github.io/labs/pillow-complete-guide) |
| 21 | **OpenCV Python Guide** | Color spaces, thresholding, edge detection & contour tracking | `Hands-on` | [Start Lab ↗](https://inferencecollective.github.io/labs/opencv-complete-guide) |
| 22 | **scikit-image Scientific Vision** | Morphology, feature extraction, skimage segmentation | `Intermediate` | [Start Lab ↗](https://inferencecollective.github.io/labs/scikit-image-complete-guide) |
| 23 | **ML 101: A Gentle Introduction** | Core machine learning paradigms, cost functions & metrics | `Foundational` | [Start Lab ↗](https://inferencecollective.github.io/labs/ml-101) |
| 24 | **Data Science Core Architecture** | Ecosystem interactions between array runtimes & models | `Foundational` | [Start Lab ↗](https://inferencecollective.github.io/labs/data-science-libraries) |
| 25 | **Before GPU Engineering** | Memory physics, memory bandwidth, FLOPs & roofline model | `Advanced` | [Start Lab ↗](https://inferencecollective.github.io/labs/before-gpu-engineering) |

---

## 🧰 Curated AI & MCP Tools Showcase

> 🌟 **Automated Directory Sync**: Search, filter by tier, and inspect all projects on **[inferencecollective.github.io/tools](https://inferencecollective.github.io/tools)**.

<a id="quick-navigation"></a>
### 🧭 Quick Navigation by Ecosystem Domain

| Domain / Stage | Core Capabilities | Project Count | Quick Jump |
| :--- | :--- | :---: | :---: |
| **🏛️ LF AI & Data Graduated** | Production-proven foundation runtimes, orchestrators & vector engines | `14 projects` | [Browse ↓](#graduated-projects) |
| **🚀 LF AI Incubating & Sandbox** | Document parsers, feature stores, declarative ML & acceleration | `36 projects` | [Browse ↓](#incubating-sandbox) |
| **⚡ Model Context Protocol (MCP)** | Agent tool servers, Claude Desktop bridges & gateway runtimes | `1,000+ tools` | [Browse ↓](#mcp-tools) |
| **🛡️ AI Security & Guardrails** | LLM red-teaming, prompt injection defense & safety scanners | `18 projects` | [Browse ↓](#ai-security) |
| **🗄️ Vector Databases & Retrieval** | High-dimensional embedding stores, ANN indexing & hybrid search | `12 projects` | [Browse ↓](#vector-dbs) |

---

<a id="graduated-projects"></a>
### 🏛️ LF AI & Data Foundation — Graduated Projects

> _Enterprise-grade, community-governed projects meeting the highest standards of production maturity._

| Project | Domain & Focus | Official Repository |
| :--- | :--- | :--- |
| **[Milvus](https://milvus.io)** | Distributed, cloud-native vector database built for massive embedding scale and sub-second similarity search. | [`milvus-io/milvus`](https://github.com/milvus-io/milvus) |
| **[ONNX](https://onnx.ai)** | Open format for representing machine learning models, enabling seamless interoperability across deep-learning frameworks. | [`onnx/onnx`](https://github.com/onnx/onnx) |
| **[Flyte](https://flyte.org)** | Highly scalable, production-grade workflow automation and ML orchestration platform for data engineering pipelines. | [`flyteorg/flyte`](https://github.com/flyteorg/flyte) |
| **[Feast](https://feast.dev)** | Open-source feature store for machine learning, operationalizing feature management across real-time and offline training. | [`feast-dev/feast`](https://github.com/feast-dev/feast) |
| **[OpenDXL](https://www.opendxl.com)** | Open Data eXchange Layer for standardized messaging across security systems and AI agents. | [`opendxl/opendxl-client-python`](https://github.com/opendxl/opendxl-client-python) |
| **[Angel ML](https://github.com/Angel-ML/angel)** | High-performance parameter server platform for distributed machine learning with billions of variables. | [`Angel-ML/angel`](https://github.com/Angel-ML/angel) |

<p align="right"><a href="#quick-navigation">↑ Back to Ecosystem Navigation</a></p>

---

<a id="incubating-sandbox"></a>
### 🚀 LF AI & Data Foundation — Incubating & Sandbox Projects

| Project | Stage | Focus & Capabilities | Repository |
| :--- | :---: | :--- | :--- |
| **[Docling](https://github.com/DS4SD/docling)** | `Sandbox` | High-accuracy document parsing, table recognition, and OCR for ingestion into RAG and LLM systems. | [`DS4SD/docling`](https://github.com/DS4SD/docling) |
| **[Ludwig](https://ludwig.ai)** | `Incubating` | Declarative deep learning framework allowing developers to train and fine-tune models through simple YAML configurations. | [`ludwig-ai/ludwig`](https://github.com/ludwig-ai/ludwig) |
| **[DeepSpeed](https://github.com/microsoft/DeepSpeed)** | `Incubating` | Deep learning optimization software suite enabling extreme-scale model training, ZeRO memory efficiency, and inference. | [`microsoft/DeepSpeed`](https://github.com/microsoft/DeepSpeed) |
| **[Marquez](https://marquezproject.ai)** | `Incubating` | Open-source metadata service for the collection, aggregation, and visualization of data lineage and job execution. | [`MarquezProject/marquez`](https://github.com/MarquezProject/marquez) |
| **[Horovod](https://github.com/horovod/horovod)** | `Incubating` | Distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet with ring-allreduce. | [`horovod/horovod`](https://github.com/horovod/horovod) |

<p align="right"><a href="#quick-navigation">↑ Back to Ecosystem Navigation</a></p>

---

<a id="ai-security"></a>
### 🛡️ AI Security, Guardrails & Red-Teaming Tools

| Project | Focus & Capabilities | Repository |
| :--- | :--- | :--- |
| **[Garak](https://github.com/NVIDIA/garak)** | LLM vulnerability scanner probing hallucination, prompt injection, jailbreaks, data leakage, and toxic output. | [`NVIDIA/garak`](https://github.com/NVIDIA/garak) |
| **[Promptfoo](https://github.com/promptfoo/promptfoo)** | Test-driven security evaluation, red-teaming, and regression testing for LLMs, prompts, and agentic workflows. | [`promptfoo/promptfoo`](https://github.com/promptfoo/promptfoo) |
| **[DeepEval](https://github.com/confident-ai/deepeval)** | Unit testing framework for LLMs evaluating hallucination, answer relevancy, G-Eval, and RAG pipelines. | [`confident-ai/deepeval`](https://github.com/confident-ai/deepeval) |
| **[NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)** | Programmable input, output, retrieval, dialog, and execution safeguards for generative AI applications. | [`NVIDIA-NeMo/Guardrails`](https://github.com/NVIDIA-NeMo/Guardrails) |
| **[PyRIT](https://github.com/microsoft/PyRIT)** | Microsoft's Python Risk Identification Tool for automated generative AI red-teaming and adversarial probing. | [`microsoft/PyRIT`](https://github.com/microsoft/PyRIT) |
| **[OpenClaw](https://github.com/inferencecollective/openclaw)** | Local autonomous AI agent framework with sandboxed execution, least-privilege tool access, and live audit logging. | [`inferencecollective/openclaw`](https://github.com/inferencecollective/openclaw) |

<p align="right"><a href="#quick-navigation">↑ Back to Ecosystem Navigation</a></p>

---

## 📝 Engineering Research & Field Notes

- 📰 [**Qwen Coder Models: Breakthrough AI at Your Fingertips**](https://inferencecollective.github.io/blog/Qwen-Coder-Models) — Deep architectural dive into Qwen 2.5 Coder 32B, local code synthesis performance, and multi-file reasoning.
- 📰 [**Model Context Protocol (MCP) Impact on SDLC**](https://inferencecollective.github.io/blog/MCP_SDLC) — How open client-server interfaces are standardizing tool calling, context injection, and automated agent development.
- 📰 [**Curated AI and Cloud-Native Tools Collection**](https://inferencecollective.github.io/blog/cloud-native-tools-collection) — Architectural comparison of production vector databases, pipeline orchestrators, and feature stores.
- 📰 [**Quick Start Guide to AI Data Foundation**](https://inferencecollective.github.io/blog/quick-start-guide) — Navigating catalogs, setting up local benchmark environments, and contributing datasets.
- 📰 [**Tool Categories and How to Contribute**](https://inferencecollective.github.io/blog/categories-and-contributing) — Taxonomy, criteria, and governance procedures for graduating projects into the foundation catalogs.
- 📰 [**Interactive Documentation & Markdown Showcase**](https://inferencecollective.github.io/blog/markdown-showcase) — Complete styling guide demonstrating KaTeX mathematical notation, syntax highlighting, and responsive diagrams.

---

## 🛡️ AI Security & Threat Intelligence Matrix

| Threat Vector | Description & Mechanism | Defensive Mitigation | Verified Tools |
| :--- | :--- | :--- | :--- |
| **Prompt Injection** | Adversarial instructions bypassing system prompts to alter model behaviors. | Runtime input classification, semantic firewalls, and dual-token boundaries. | `Promptfoo`, `Garak`, `NeMo Guardrails` |
| **Training Data Poisoning** | Malicious data injection into training corpora or retrieval vector indices. | Strict data provenance, perceptual hashing, and embedding anomaly detection. | `DeepEval`, `Cleanlab` |
| **Model Inversion / Extraction** | Query attacks designed to reconstruct private training weights or sensitive inputs. | Differential privacy guarantees, rate limiting, and output fuzzing. | `PyRIT`, `OpenClaw` |
| **Tool / MCP Hijacking** | Exploiting agent tool calling to execute unauthorized system commands or read tokens. | Strict JSON-schema parameter validation, human-in-the-loop gating, and ephemeral sandboxing. | `OpenShell`, `Snyk Agent Scan` |
| **Data Exfiltration** | Coerced disclosure of API credentials, keys, or sensitive internal documents. | Real-time output DLP filters, regex pattern guards, and token masking. | `Gitleaks`, `TruffleHog` |

---

## 💻 Run Locally & Development Workflow

### Prerequisites

- **Elixir & Erlang**: Elixir `~> 1.17`, Erlang/OTP `~> 27`
- **Python**: Python `~> 3.11` (for test suites and ML labs)
- **Node.js**: (optional, asset bundles are handled via hex packages `esbuild` & `tailwind`)

### 1. Setup Dependencies
```sh
# Fetch and compile Elixir packages, setup Tailwind v4 and esbuild
mix setup
```

### 2. Start Local Development Server
```sh
# Boots Phoenix endpoint on http://localhost:4000 with live reload
mix phx.server
```

### 3. Run Test Suites & Quality Verification
```sh
# Run Elixir ExUnit test suite
mix test

# Run Python lab execution test suite (35 integration tests)
python3 -m pytest -q

# Run full precommit suite (warnings-as-errors, format check, and tests)
mix precommit
```

### 4. Static Site Export (GitHub Pages)
```sh
# Build minified assets
mix assets.deploy

# Export all 39 dynamic routes into static HTML and digest bundles in _site/
mix aidatafoundation.export --output _site
```

The output directory `_site/` is automatically deployed to GitHub Pages on every push to `main` via `.github/workflows/pages.yml`.

---

## 🤝 Contributing to Inference Collective

We welcome contributions from AI researchers, data engineers, and security practitioners:

1. **Submit an AI Tool or Framework**: Add your project details to `priv/static/aidatafoundation/tools.js` with category, tags, and GitHub metadata.
2. **Submit Benchmark Data**: Add verified benchmark scores and VRAM metrics to `priv/static/aidatafoundation/data/model.json`.
3. **Contribute a Lab**: Submit an interactive terminal-first learning guide in `content/labs/` with runnable Python source code.
4. **Report Security Vulnerabilities**: Review our disclosure policy on the [Security page](https://inferencecollective.github.io/security).

```sh
# Fork repository, create feature branch, and submit PR
git checkout -b feat/add-new-mcp-tool
mix precommit
git push origin feat/add-new-mcp-tool
```

---

## 📜 License & Community

- **License**: Released under the [Apache 2.0 License](LICENSE).
- **Organization**: [github.com/inferencecollective](https://github.com/inferencecollective)
- **Website**: [inferencecollective.github.io](https://inferencecollective.github.io/)
- **Maintained By**: The Inference Collective community and contributors.
