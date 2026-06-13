# NeuroDSL — A Dynamic Computational Graph Framework

[![arXiv](https://img.shields.io/badge/arXiv-2606.XXXXX-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2606.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.10%2B-purple?style=flat-square&logo=julia)](https://julialang.org)
[![GPU](https://img.shields.io/badge/GPU-CUDA-green?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

**NeuroDSL** is a deep learning framework that treats the computational graph as a **mutable, living entity** — not a static trace.  
It lets you **modify the graph during training**, recomputes only what’s needed, and guarantees memory efficiency through **provably optimal interval coloring**.

---

## ✨ Key Innovations

- 🔄 **Dynamic DAG** – add, remove, or fuse operations on the fly  
- 💾 **Optimal Memory Planning** – up to **2–6× less VRAM** than PyTorch (mathematically guaranteed)  
- ⚡ **Self‑Optimizing** – automatic operator fusion (+37% CPU / +54% GPU), incremental backward (up to 89% less recomputation)  
- 👁️ **Reactive Callbacks** – nodes that respond on their own (early stopping, LR adjustment)  
- 🎥 **Live Viewer** – watch your network think: forward in blue, backward in red, loss curve dancing  
- 🖥️ **Multi‑GPU** – near‑linear speedup with periodic weight sync  
- 📐 **Category‑Theoretic Foundations** – rigorous mathematical underpinnings  

---

## 📊 Performance Highlights

| Metric | NeuroDSL | vs PyTorch / Flux |
|--------|----------|-------------------|
| Transformer block (dim=1024) | 5.96 ms | 1.03× PyTorch, 1.42× Flux |
| Peak GPU memory (dim=1024) | 34.61 MiB | **2.15× less** than PyTorch |
| Iris classification | 100% test accuracy | — |
| Operator fusion | +37% CPU / +54% GPU | — |
| Incremental backward | 24–89% gain | — |
| Early stopping | 97.2% fewer iterations | — |
| Multi‑GPU (2 GPUs) | 1.9× speedup | vs single GPU |

---

## 📦 Installation

```bash
git clone https://github.com/nevermind78/NeuroDSL.jl.git
cd NeuroDSL.jl
julia --project -e 'import Pkg; Pkg.instantiate()'
```

Then in Julia:

```julia
using NeuroDSL
```

---

## 🧪 Quick Start

```julia
using NeuroDSL

# Build a dynamic graph
g = NeuroDSL.JuliusGraph()
NeuroDSL.set!(g, :x, randn(Float32, 4, 4))
NeuroDSL.set!(g, :W, randn(Float32, 4, 4); is_param=true)
NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:y, [:x, :W], :matmul; attrs=Dict(:trans_b=>true)))
NeuroDSL.addrule!(g, NeuroDSL.GraphRule(:loss, [:y], :sum_matrix))

# Forward & backward
ctx = NeuroDSL.CtxStore()
NeuroDSL.demand!(g, :loss; ctx_store=ctx)
NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx)

# Visualize
log = NeuroDSL.ExecutionLog()
NeuroDSL.demand!(g, :loss; ctx_store=ctx, log=log)
NeuroDSL.backward_graph!(g, :loss; ctx_store=ctx, log=log)
NeuroDSL.save_interactive_graph(g, log, "demo.html"; title="My First NeuroDSL Graph")
```

Open `demo.html` in your browser (via a local server) and step through the computation.

---

## 🎥 See It in Action


<video width="100%" controls>
  <source src="html/NeuroDSL.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

> *“Imagine a graph that can think on its own, respond on its own, and recompute only what’s needed.”*

---

## 🧠 Design Philosophy

NeuroDSL is not just another deep learning library. It’s a **live DAG engine**:

- **Mutable** — the graph can be rewired at any time  
- **Lazy** — only invalid subgraphs are recomputed  
- **Memory‑aware** — optimal buffer sharing via interval coloring  
- **Self‑optimizing** — fusion, incremental backprop, and reactive triggers work out of the box  

The interactive viewer makes these dynamic behaviors **visible** — ideal for debugging, education, and research.

---

## 📂 Repository Structure

- `src/` — the core NeuroDSL module  
- `test/` — unit and integration tests  
- `benchmarks/` — reproducible experiments  
- `figures/` — plots and diagrams for the paper  
- `Article.md` — the LaTeX manuscript  
- `docs/` — additional documentation  

---

## 📖 Citation

```bibtex
@article{neurodsl2026,
  title   = {NeuroDSL: A Dynamic Computational Graph Framework with Optimal Memory Planning},
  author  = {Your Name},
  journal = {arXiv preprint arXiv:2606.XXXXX},
  year    = {2026}
}
```

---

## 🏆 Inspiration

NeuroDSL was inspired by the early work of [Julius Technology](https://juliustechco.github.io/JuliusGraph/dev/) on dynamic graph engines. We extend their ideas with formal memory planning, automatic optimizations, and an interactive live viewer.

---

## 📬 Contact

- **GitHub:** [@nevermind78](https://github.com/nevermind78)  
- **Email:** [khemais.abdallah@isitc.u-sousse.tn](mailto:khemais.abdallah@isitc.u-sousse.tn)  
- **LinkedIn:** [LinkedIn](https://www.linkedin.com/in/khemais-abdallah/)

---

<p align="center">
  <br>
  <img src="https://img.shields.io/badge/Made%20with-Julia-9558B2?style=for-the-badge&logo=julia">
  <img src="https://img.shields.io/badge/Powered%20by-NeuroDSL-blueviolet?style=for-the-badge">
  <br><br>
  <i>“It’s only in the living equations of a dynamic graph that any logic can truly be found.”</i>
</p>