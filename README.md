# Genetic Methuselah Finder

This project is a **Genetic Algorithm** designed to discover "Methuselahs" in **Conway's Game of Life**. A Methuselah is a specific initial pattern of live cells that evolves for many generations before stabilizing or repeating, often producing complex and interesting behaviors along the way.

The project uses evolutionary principles such as **selection**, **crossover**, and **mutation** to explore the vast space of possible initial configurations and find the most promising candidates.

---

## Features

- **Genetic Algorithm Core**:
  - Individuals are represented as 2D grids of binary values (alive/dead).
  - The fitness function evaluates grids based on longevity, population dynamics, and periodic behavior.
  - Mutation rates adapt if progress stagnates.

- **Game of Life Simulation**:
  - Handles periodic, stable, and extinction outcomes.
  - Tkinter-based GUI for visualizing the best solution.

- **Genetic Operators**:
  - **Tournament Selection** for parent choice.
  - Single-point crossover to create offspring.
  - Mutation flips cells based on population conditions.

- **Fitness Over Generations Graph**:
  - Uses Matplotlib to show fitness improvement over generations.

---

## Requirements

- Python 3.9+  
- Libraries:
  - `tkinter` (usually included with Python)
  - `matplotlib`

Install `matplotlib` if needed:
```bash
pip install matplotlib
