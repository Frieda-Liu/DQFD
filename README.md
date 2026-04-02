# 🔋 Multi-Agent EV Routing & Charging Optimization with RL

**A Deep Reinforcement Learning Framework for Smart City Logistics**

> **Live Demo: [Explore the Interactive Dashboard here!](https://ps8jhguwtc4ad5mx4qpf6b.streamlit.app)**

## Overview

This project is a Multi-Agent Reinforcement Learning (MARL) evaluation platform designed to solve **Electric Vehicle (EV) routing and charging scheduling** problems in urban environments. The simulation is built on real-world geographic data from **London, Ontario**, utilizing Uber’s **H3 Hexagonal Hierarchical Spatial Index** for precise spatial modeling.

The core engine uses a **Deep Q-Network (DQN)** architecture integrated with **Expert Guidance** and dynamic **Weather Factors** to train agents to navigate safely while maintaining optimal State of Charge (SoC) levels.

---

## Key Features

- **H3 Urban Grid**: Realistic hexagonal modeling of London, ON.
- **Multi-Agent Coordination**: Supports 1-20 agents with congestion awareness.
- **EV Infrastructure**: Integrated **L2/L3 Charging Stations** for energy recovery.
- **Autonomous SoC Logic**: Real-time battery management & station seeking.
- **Weather & Consumption**: Dynamic energy impact based on weather conditions (0.8x - 2.0x).
- **Interactive UI**: Full Streamlit dashboard for evaluation & trajectory replay.

---

## Technical Stack

- **Deep Learning**: PyTorch
- **RL Environment**: Custom OpenAI Gym-compatible interface (`mutilEnv.py`)
- **Spatial Data**: H3-py, GeoPandas, Shapely
- **Map Rendering**: Contextily (OpenStreetMap), Matplotlib
- **Web Framework**: Streamlit

---

## Methodology

### State Space

The agent perceives a 20-dimensional observation vector, including: current relative coordinates, distance to goal, distance to nearest chargers, current SoC, and local neighbor density.

### Reward Function

The reward structure balances multiple objectives:
$$R = r_{goal} + r_{energy} + r_{safety} + r_{efficiency}$$

- **Goal Achievement**: Significant positive reward for reaching the destination.
- **Energy Penalty**: Deductions for battery depletion or inefficient movement.
- **Safety & Constraints**: Penalties for off-road movement or exceeding maximum steps (Timeout).

---

## Installation & Quick Start

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/Frieda-Liu/DQFD.git](https://github.com/Frieda-Liu/DQFD.git)
   cd DQFD
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Install dependencies**:
   ```bash
   streamlit run Train/app.py
   ```

---

## Project Structure

```text
.
├── Train/
│   ├── app.py              # Main Streamlit dashboard application
│   ├── models/             # Folder containing trained .pth model files
│   ├── mutilEnv.py         # Custom Multi-Agent Hex-Traffic Environment
│   └── mutilDqfsAgent.py   # ExpertDQN/DqfD Agent implementation
├── data/                   # (Optional) Static map data or road networks
├── requirements.txt        # List of required Python packages
└── README.md               # Project documentation
```

---

## 🎓 Academic Context

**Institution:** Western University (UWO)  
**Project Team:** Siyi Liu  
**Supervisors:** Prof. Tang
