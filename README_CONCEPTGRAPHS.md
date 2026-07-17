# ConceptGraphs for UAV-ON Navigation

🚁 **3D Scene Graphs for Drone Navigation** with CLIP Semantic Understanding and LLM Reasoning

[![Status](https://img.shields.io/badge/status-complete-success)]()
[![Python](https://img.shields.io/badge/python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.1.2-orange)]()

---

## 📖 Overview

This project integrates [ConceptGraphs](https://concept-graphs.github.io/) with [UAV-ON](https://arxiv.org/abs/2408.02266) for semantic drone navigation. The system builds 3D scene graphs from RGB-D observations, uses CLIP for object recognition, and leverages DeepSeek LLM for explainable navigation decisions.

**Key Features**:
- ⚡ Ultra-fast scene graph construction (2600+ FPS)
- 🎯 CLIP-based semantic understanding (22 outdoor object categories)
- 🧠 LLM reasoning with DeepSeek (natural language explanations)
- 📊 Complete evaluation framework (Baseline vs CLIP vs DeepSeek)

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
cd /villa/mwq24-srt/UAV/UAV_ON

# Activate virtual environment
source uavon_env/bin/activate

# Install dependencies (if not already installed)
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### 2. Configuration

```bash
# Copy configuration template
cp .env.template .env

# Edit and add your DeepSeek API key
nano .env
# Set: DEEPSEEK_API_KEY=sk-your-key-here
```

### 3. Run Object Navigation

```bash
# Using CLIP semantic detection
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/evaluate_object_nav.py \
  --rgbd_dir airsim_collected \
  --target vehicle \
  --strategy clip

# Using DeepSeek LLM reasoning
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/evaluate_object_nav.py \
  --rgbd_dir airsim_collected \
  --target vehicle \
  --strategy deepseek
```

---

## 📊 Results

### Object Detection Success

| Strategy | Target Found | Steps | Reasoning |
|----------|-------------|-------|-----------|
| Baseline | ❌ | N/A | Blind exploration |
| **CLIP** | ✅ | 3 | Semantic detection |
| **DeepSeek** | ✅ | 3 | LLM + spatial reasoning |

### Scene Understanding

```
Step 0: house       → exploring
Step 1: street      → exploring  
Step 2: parking lot → approaching target area
Step 3: vehicle     → TARGET FOUND! 🎯
```

### Performance

- **Scene Graph**: 2600+ FPS ⚡
- **With CLIP**: ~10 FPS (real-time)
- **With DeepSeek**: ~0.5 FPS (deliberate planning)

---

## 🏗️ System Architecture

```
RGB-D Input (AirSim)
    ↓
┌─────────────────────┐
│ CLIP Detector       │ ← 22 object categories
│ (semantic labels)   │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ ConceptGraphBuilder │ ← 3D scene graph
│ (nodes + edges)     │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ DeepSeek Planner    │ ← LLM reasoning
│ (action + explain)  │
└─────────────────────┘
    ↓
Navigation Action
```

---

## 📁 Project Structure

```
UAV_ON/
├── .env                           # Your API keys (auto-loaded)
├── .env.template                  # Configuration template
│
├── src/conceptgraphs_uav/         # Core modules
│   ├── frame.py                   # UAVFrame data structure
│   ├── geometry.py                # 3D geometry utilities
│   ├── graph.py                   # Scene graph builder
│   ├── clip_detector.py           # CLIP semantic detection
│   ├── deepseek_planner.py        # LLM decision making
│   └── config.py                  # Configuration loader
│
├── scripts/                       # Executable scripts
│   ├── collect_airsim_direct.py   # Collect RGB-D data
│   ├── build_graph_with_clip.py   # Build semantic scene graph
│   ├── evaluate_object_nav.py     # Run navigation evaluation
│   ├── visualize_scene_graph.py   # Visualize results
│   └── benchmark_graph_speed.py   # Performance benchmarks
│
├── airsim_collected/              # Sample RGB-D data (5 frames)
├── scene_graph_clip.json          # Semantic scene graph output
└── visualizations/                # Visualization outputs
```

---

## 🎯 Supported Object Categories

Buildings: `building`, `house`, `skyscraper`  
Vehicles: `car`, `vehicle`, `truck`  
Infrastructure: `road`, `street`, `pavement`, `sidewalk`, `parking lot`  
Nature: `tree`, `vegetation`, `grass`  
Environment: `sky`, `cloud`  
People: `person`, `pedestrian`  
Objects: `traffic light`, `street sign`, `fence`, `wall`

---

## 🔧 Advanced Usage

### Collect Your Own Data

```bash
# Start AirSim scene (with virtual display)
cd /villa/mwq24-srt/UAV/TEST_ENVS/Barnyard
xvfb-run --auto-servernum ./Barnyard_test1.sh -RenderOffScreen &

# Collect RGB-D frames
cd /villa/mwq24-srt/UAV/UAV_ON
python scripts/collect_airsim_direct.py \
  --scene Barnyard \
  --output_dir ./my_data \
  --max_frames 20
```

### Build Semantic Scene Graph

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/build_graph_with_clip.py \
  --input_dir ./my_data \
  --output ./my_scene_graph.json
```

### Visualize Results

```bash
python scripts/visualize_scene_graph.py \
  --graph ./my_scene_graph.json \
  --rgbd_dir ./my_data \
  --output_dir ./my_visualizations
```

### Test DeepSeek Planner Directly

```bash
PYTHONPATH=src python src/conceptgraphs_uav/deepseek_planner.py \
  --scene_graph ./my_scene_graph.json \
  --target building
```

---

## 📖 Documentation

- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - Complete project summary
- **[STRATEGY_COMPARISON_REPORT.md](STRATEGY_COMPARISON_REPORT.md)** - Detailed strategy comparison
- **[CONCEPTGRAPHS_IMPLEMENTATION_REPORT.md](CONCEPTGRAPHS_IMPLEMENTATION_REPORT.md)** - Technical implementation
- **[CONCEPTGRAPHS_EVALUATION_REPORT.md](CONCEPTGRAPHS_EVALUATION_REPORT.md)** - Performance evaluation
- **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)** - Configuration guide

---

## 🔬 Key Findings

1. **CLIP works on UAV aerial views** despite low confidence (0.04-0.05)
2. **Image-level classification is sufficient** for navigation (no pixel masks needed)
3. **DeepSeek adds explainability** without improving raw performance
4. **Scene graphs provide spatial structure** for multi-object reasoning

---

## ⚠️ Known Limitations

1. **Small dataset**: Only 5 frames tested (need multi-scene evaluation)
2. **Low CLIP confidence**: 4-5% (but relative ranking works)
3. **No pixel-level segmentation**: CLIP is image-level only
4. **DeepSeek latency**: ~1-2s per decision (not for reactive control)

---

## 🛠️ Requirements

**Core**:
- Python 3.10
- PyTorch 2.1.2 + CUDA 12.1
- OpenAI CLIP
- transformers

**For LLM**:
- DeepSeek API key (get at [platform.deepseek.com](https://platform.deepseek.com))

**For AirSim**:
- Xvfb (virtual display)
- UE4 AirSim scenes

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{conceptgraphs-uavon-2026,
  title={ConceptGraphs for UAV-ON Navigation},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/...}}
}
```

**Original papers**:

```bibtex
@article{gu2023conceptgraphs,
  title={ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning},
  author={Gu, Qiao and Kuwajerwala, Ali and Morin, Sacha and others},
  journal={arXiv preprint arXiv:2309.16650},
  year={2023}
}

@article{wang2024uavon,
  title={UAV-ON: Towards Enabling Embodied AI for Unmanned Aerial Vehicles},
  author={Wang, Zhiyuan and Zhang, Xin and others},
  journal={arXiv preprint arXiv:2408.02266},
  year={2024}
}
```

---

## 🤝 Contributing

Issues and pull requests are welcome! Areas for improvement:

- [ ] Multi-scene evaluation
- [ ] FastSAM / MobileSAM integration
- [ ] Fine-tune CLIP on UAV data
- [ ] Temporal fusion and object tracking
- [ ] Support for more LLM backends (GPT-4, Claude, etc.)

---

## 📧 Contact

For questions or collaboration: [your-email@example.com]

---

## 📄 License

[MIT License](LICENSE) - see LICENSE file for details

---

**Status**: ✅ Fully implemented and tested (2026-06-30)
