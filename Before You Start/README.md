# Before You Start: AI Data Foundation Labs & Workshops

Welcome to the **AI Data Foundation** hands-on labs and workshops track. Before diving into the practical exercises, ensure your development environment is properly configured.

---

## 1. Prerequisites Checklist

To follow along with all labs, you will need:

| Requirement | Recommended Version / Tool | Purpose |
| :--- | :--- | :--- |
| **Python** | `>= 3.11` (managed via `uv`) | Core runtime for running models, benchmarks, and pipelines |
| **Package Manager** | [`uv`](https://github.com/astral-sh/uv) (Astral) | Lightning-fast Python package installation and environment isolation |
| **Container Engine** | Docker / Docker Compose / OrbStack | Running vector databases (Milvus, Weaviate, Qdrant) and local inference engines |
| **Local LLM Engine** | [Ollama](https://ollama.ai) or [vLLM](https://github.com/vllm-project/vllm) | Local zero-cost model testing and offline token generation |
| **Git** | `>= 2.40` | Version control and submodules |

---

## 2. Environment Setup (Recommended: `uv`)

We strongly recommend using `uv` instead of standard `venv` or `conda` for instant dependency resolution:

```bash
# Install uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version

# Create and activate an isolated Python 3.12 virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate
```

---

## 3. Core Libraries Installation

Install the foundation runtime dependencies:

```bash
uv pip install \
  torch torchvision torchaudio \
  numpy pandas scikit-learn \
  transformers accelerate datasets \
  langchain langchain-community \
  pymilvus weaviate-client \
  marimo jupyterlab
```

---

## 4. Hardware & GPU Accelerators

- **Apple Silicon (M1/M2/M3/M4)**: PyTorch MPS acceleration is supported out of the box (`device = "mps"`).
- **NVIDIA GPU (CUDA)**: Ensure CUDA 12.1+ and cuDNN are installed (`device = "cuda"`).
- **CPU-Only**: Small quantized models (e.g., GGUF via Ollama or 4-bit ONNX Runtime) will run efficiently without dedicated GPU hardware.

---

## 5. API Keys Configuration (Optional)

Create a `.env` file in your workspace for external model providers:

```bash
# Optional external providers
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
HUGGINGFACE_TOKEN="hf_..."
```

---

## 6. Guides & Labs in this Series

- [01. Data Science & ML Libraries Architecture](data-science-libraries.md): Ecosystem layers, Pele/Mermaid diagrams, mental models, GBDT comparisons, and 2026 developer trends.
- [02. NumPy in Depth: Memory Architecture, Strides & Tensor Contractions](02-numpy-in-depth.md): Raw contiguous C buffers, byte-offset formulas, zero-copy views vs copies, SIMD benchmarks, broadcasting algebra (stride=0 trick), and transformer self-attention with `np.einsum`.
- [03. Before GPU Engineering: Hardware Architecture, Memory Physics & The Roofline Model](03-before-gpu-engineering.md): First-principles guide to CPU vs GPU silicon die allocation, memory latency pyramids (Registers to HBM3), arithmetic intensity, SIMT warps, branch divergence, and why LLM decoding is memory-bandwidth bound.
- [04. SciPy Complete Guide: Beginner to Intermediate Scientific Python](04-scipy-complete-guide.md): Optimization, linear algebra, integration, ODEs, interpolation, hypothesis testing, FFT signal processing, sparse arrays, and spatial KDTree.
- [05. Pandas Complete Guide: From DataFrames to SQL Joins and Analytics](05-pandas-complete-guide.md): DataFrame/Series anatomy, loc vs iloc, CSV cleaning, GroupBy aggregations, SQL-style joins, time series resampling, pivot/melt, SQLite comparison, and Copy-on-Write.
- [06. Matplotlib Complete Guide: Python Charts, Figures & Five Hands-On Labs](06-matplotlib-complete-guide.md): Figure/Axes anatomy, lines, scatter, bar charts, distributions, subplot mosaics, tick formatting, heatmaps, and publication export.
- [07. Seaborn Complete Guide: Statistical Visualizations and Five Python Labs](07-seaborn-complete-guide.md): Statistical graphics, distribution trio (hist/KDE/ECDF), categorical variance, correlation matrices, FacetGrid small multiples, and bootstrap uncertainty.
- [08. Scikit-learn Complete Guide: Beginner to Intermediate](08-scikit-learn-complete-guide.md): Classical machine learning, Estimator API, stratified splits, classification metrics, Ridge regression diagnostics, ColumnTransformer pipelines, GridSearchCV tuning, KMeans, and PCA.
- [09. XGBoost Complete Guide: 10 Lessons + 5 Python Labs](09-xgboost-complete-guide.md): Gradient-boosted decision trees, objective optimization, DMatrix, early stopping without data leakage, hist vs exact tree methods, native categoricals, and JSON model deployment.
- [10. LightGBM Complete Learning Guide](10-lightgbm-complete-guide.md): Gradient-boosted decision trees, histogram binning, leaf-wise vs level-wise growth, LGBMClassifier, LGBMRegressor, native Dataset and Booster, leakage-free early stopping callbacks, native pandas categorical features, GridSearchCV tuning, and Booster text model serialization.
- [11. CatBoost Complete Guide: Python Tutorial, Pool, Categories & Five ML Labs](11-catboost-complete-guide.md): Gradient-boosted decision trees, oblivious/symmetric decision trees, native Pool container, robust string categorical feature encoding, missing value imputation, leakage-free early stopping, manual StratifiedKFold CV tuning, and native CBM model serialization.
- [12. Complete Pillow Guide — Python Image Processing, Examples and Labs](12-pillow-complete-guide.md): Raster image processing, format conversion across PNG/JPEG/WebP, proportional resizing with contain, fit and pad, RGBA alpha compositing, spatial filters and ImageEnhance, ImageDraw typography, EXIF normalization, batch processing, and defensive upload validation.
- [13. OpenCV Python Complete Guide — 10 Lessons and 5 Computer Vision Labs](13-opencv-complete-guide.md): Computer vision with OpenCV and NumPy, BGR color spaces, Gaussian and Canny edge detection, contour hierarchy and morphology, homography and four-point perspective transforms, and real-time video frame object tracking.
- [14. scikit-image Complete Guide: Scientific Image Processing in Python](14-scikit-image-complete-guide.md): Scientific image analysis, RGB/gray conversions, floating-point dynamic range normalization, Gaussian and Canny filters, Otsu thresholding, footprint morphology, connected components regionprops morphometry, distance transform watershed segmentation, and 3D volumetric analysis.
- [15. PyTorch Complete Guide — Tensors, Autograd, Neural Networks & Transformer Basics](15-pytorch-complete-guide.md): Deep learning foundations, tensor memory layout and striding, dynamic autograd computation graphs, nn.Module layer construction, mini-batch DataLoader pipelines, 2D CNN image classification, token embedding NLP with masked mean pooling, and causal multihead self-attention transformer blocks.
- [16. TorchVision Complete Guide — 10 Lessons and 5 Hands-on Labs](16-torchvision-complete-guide.md): Computer vision engineering with TorchVision, CHW tensor representations, ImageFolder dataset hierarchies, modern transforms.v2 compositions, data augmentation without label leakage, make_grid visual debugging, TinyVisionCNN training and validation checkpointing, Model Zoo weight inspection, and TVTensors synchronized bounding boxes and masks.
- [17. TensorFlow Complete Guide — Beginner to Deployment](17-tensorflow-complete-guide.md): Deep learning systems with TensorFlow 2.x and Keras 3, static vs dynamic computational graphs, tf.Tensor and NHWC memory layouts, tf.GradientTape automatic differentiation, Keras Sequential and Functional architectures, high-performance tf.data pipelines (batch, prefetch, map), early stopping and validation splits, 2D CNN classification on synthetic shapes, SavedModel serialization with serving signatures, and LiteRT (TFLite) mobile and edge deployment with dynamic-range quantization.
- [18. Keras 3 Complete Guide — From Python Tensors to Multi-Backend AI](18-keras-complete-guide.md): Deep learning architecture with Keras 3 multi-backend execution across PyTorch, JAX, and TensorFlow, symbolic KerasTensor vs concrete backend tensors, backend-agnostic mathematical operations with keras.ops, Sequential vs Functional model graphs, reliable 3-way dataset partitioning (train/val/test) with EarlyStopping, 2D CNN classification on channels-last synthetic vision fixtures, token embeddings and multi-head self-attention, backend-neutral custom layers with @register_keras_serializable, safe .keras model persistence, and deployment export pathways.

---

## 7. Labs Navigation

All hands-on labs are available under:
- Web: [aidatafoundation.github.io/labs](https://aidatafoundation.github.io/labs)
- Markdown sources: `priv/static/aidatafoundation/labs/`

