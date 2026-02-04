# Feature Manifolds

## Repository Structure

```
data/           Synthetic linebreaking datasets
figures/        Output plots (PCA, t-SNE, UMAP, inner products)
model/          Dynamical system for sphere particle simulation
src/            Inference and analysis scripts
```

## Synthetic Linebreaking Dataset

**Location**: `data/synthetic_linebreak_k{N}.txt`

**Line widths**: k ∈ {20, 30, 40, 50, 60, 70, 80, 90, 100}

**Format**:
```
EXAMPLE 01
A small band of neighbors argued late into the
night about what a fair nation could be. They
distrusted crowns, yet feared chaos, so they
designed limits, votes, and written rules.

EXAMPLE 02
...
```

**Generation procedure**:
1. Remove all newlines from source prose
2. Insert newline at last word boundary ≤ k characters
3. Repeat until text exhausted

**Constraint**: Lines never exceed k characters; break at last space ≤ k

**Size**: 10 prose examples per file

## Key Scripts

**`model/dynamical_model.py`** — `SphereDynamics` class: N particles on (n-1)-sphere with attraction/repulsion forces based on index distance. Params: `n_particles`, `n_dimensions` (3-8), `zone_width`, `topology` ('circle'|'interval').

**`visualize_manifold.py`** — Runs sphere dynamics simulation, outputs PCA/UMAP/t-SNE projections and inner product heatmaps to `figures/`.

**`src/inference.py`** — Single-prompt newline prediction test using gemma-3-1b-pt.

**`src/newline_prob.py`** — Computes newline log-prob and top-1 accuracy across line positions. Outputs figure matching paper style.

**`src/playground.ipynb`** — Miscellaneous checks/experiments.
