# feature-manifolds

Simulation and visualization of N particles on an $(n-1)$-sphere evolving under attraction/repulsion dynamics, with dimensionality reduction analysis.

## File Structure

```
manifold-vis/
├── model/
│   └── dynamical_model.py      # Core dynamical system implementation
├── visualize_manifold.py       # Visualization functions (PCA, UMAP, t-SNE)
├── figures/                    # Generated visualization outputs
│   ├── pca/                   # PCA projections
│   ├── umap/                  # UMAP projections
│   ├── tsne/                  # t-SNE projections
│   └── inner_product/         # Inner product heatmaps
├── requirements.txt           # Python dependencies
└── README.md
```

## Core Components

### `model/dynamical_model.py`
Implements `SphereDynamics` class that simulates N particles on a unit $(n-1)$-sphere with:
- **Attractive forces**: Between particles within index distance $\leq$ `zone_width`
- **Repulsive forces**: Between particles with index distance $>$ `zone_width`
- **Constraint**: All particles remain on the unit sphere via normalization

**Key methods:**
- `__init__()`: Initialize system with particles uniformly distributed on sphere
- `simulate(n_steps)`: Run simulation, returns trajectory and velocity history
- `get_inner_product_matrix()`: Compute pairwise inner products $\langle x_i, x_j \rangle$

### `visualize_manifold.py`
Three visualization functions using different dimensionality reduction techniques:

1. **`analyze_manifold_pca()`**: Uses PCA for linear dimensionality reduction
   - Generates 3D scatter plot with variance explained
   - Creates inner product heatmap
   - Saves to `figures/pca/` and `figures/inner_product/`

2. **`analyze_manifold_umap()`**: Uses UMAP for non-linear manifold learning
   - Preserves local structure better than PCA
   - Saves to `figures/umap/`

3. **`analyze_manifold_tsne()`**: Uses t-SNE for non-linear dimensionality reduction
   - Emphasizes local neighborhoods
   - Saves to `figures/tsne/`

## Key Parameters

### Dynamical System (`SphereDynamics`)
- **`n_particles`** (int, default=100): Number of particles on sphere
- **`n_dimensions`** (int, default=6): Ambient dimension $\in \{3,4,5,6,7,8\}$; creates $(n-1)$-sphere
- **`zone_width`** (float, default=5.0): Attractive zone width $w$; controls force transition
- **`topology`** (str, default='circle'): Index distance topology $\{\text{'circle'}, \text{'interval'}\}$
- **`dt`** (float, default=0.01): Time step size
- **`damping`** (float, default=0.95): Velocity damping coefficient $\alpha$
- **`damping_linear`** (float, default=0.05): Linear damping in force equation

### Visualization Functions
- **`n_steps`** (int): Number of simulation steps (e.g., 100000 for convergence)
- All dynamical parameters above

## Force Law

$$
F_{ij} = \begin{cases}
\frac{1 - (d_{ij} - 1)/2}{r_{ij}} \hat{r}_{ij} & \text{if } d_{ij} \leq w \text{ (attractive)} \\
-\frac{\min(5, 1/r_{ij})}{r_{ij}} \hat{r}_{ij} & \text{if } d_{ij} > w \text{ (repulsive)}
\end{cases}
$$

where:
- $d_{ij}$: Index distance between particles $i$ and $j$
- $r_{ij}$: Euclidean distance $\|x_j - x_i\|$
- $\hat{r}_{ij}$: Unit direction vector $(x_j - x_i) / r_{ij}$

## Results

### Generated Visualizations

Each function produces:
- **3D scatter plots**: Particles colored by index (HSV colormap shows topology)
- **Naming convention**: `{method}_plot_n{N}_dim{d}_steps{s}_w{w}_{topology}.png` where $N$ = particles, $d$ = dimensions, $s$ = steps, $w$ = zone width

**PCA additionally generates:**
- Inner product heatmap showing pairwise correlations $\langle x_i, x_j \rangle$

### Typical Observations
- **Circle topology**: Particles form ordered structures respecting index ordering
- **PCA**: Captures global variance; linear projection
- **UMAP**: Preserves local manifold structure; reveals intrinsic geometry
- **t-SNE**: Emphasizes cluster separation; good for local neighborhoods
- **Convergence**: ~10,000 steps typically needed for stable configurations

## Usage

```python
from visualize_manifold import analyze_manifold_pca, analyze_manifold_umap, analyze_manifold_tsne

# Run simulation and generate PCA visualization
trajectory, velocities, inner_prod, positions_3d = analyze_manifold_pca(
    n_particles=100,
    n_dimensions=5,      # Creates 4-sphere
    n_steps=100000,
    zone_width=5.0,
    topology='circle'
)

# Or use UMAP
analyze_manifold_umap(n_particles=100, n_dimensions=5, n_steps=100000)

# Or use t-SNE
analyze_manifold_tsne(n_particles=100, n_dimensions=5, n_steps=100000)
```

## Installation

Install dependencies using `uv`:

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux

# Install requirements
uv pip install -r requirements.txt
```
