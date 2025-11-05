from model.dynamical_model import SphereDynamics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from pathlib import Path


def analyze_manifold_pca(n_particles: int = 100, n_dimensions: int = 6, n_steps: int = 100, zone_width: float = 5.0, topology: str = 'circle'):
    # create dynamical system with default parameters
    system = SphereDynamics(n_particles, n_dimensions, zone_width, topology)
    
    # run simulation for 100 time steps
    trajectory, velocity_history = system.simulate(n_steps=n_steps)

    # compute inner product matrix
    inner_products = system.get_inner_product_matrix()
    print(f"\nInner product matrix shape: {inner_products.shape}")
    # print(f"Diagonal values (should be ~1): {np.diag(inner_products)[:5]}")
    
    # get final positions after simulation
    final_positions = system.positions  # shape: (n_particles, n_dimensions)
    
    # perform PCA to reduce to 3D for visualization
    pca = PCA(n_components=3)
    positions_3d = pca.fit_transform(final_positions)
    
    # create output directories if they don't exist
    pca_output_dir = Path("figures/pca")
    pca_output_dir.mkdir(parents=True, exist_ok=True)
    
    heatmap_output_dir = Path("figures/inner_product")
    heatmap_output_dir.mkdir(parents=True, exist_ok=True)
    
    # create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # color points by their index (representing position along topology)
    colors = np.arange(n_particles)
    scatter = ax.scatter(
        positions_3d[:, 0], 
        positions_3d[:, 1], 
        positions_3d[:, 2],
        c=colors,
        cmap='hsv',
        s=50,
        alpha=0.7,
        edgecolors='k',
        linewidth=0.5
    )
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_zlabel('PC3', fontsize=12)
    ax.set_title(f'PCA Projection of {n_particles} Particles on {n_dimensions-1}-Sphere\n'
                 f'After {n_steps} Steps (Topology: {topology}, Zone Width: {zone_width})',
                 fontsize=14, pad=20)
    
    # add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Particle Index', fontsize=11)
    
    # add variance explained
    var_explained = pca.explained_variance_ratio_
    ax.text2D(0.02, 0.98, 
              f'Variance explained: {var_explained[0]:.2%}, {var_explained[1]:.2%}, {var_explained[2]:.2%}',
              transform=ax.transAxes, fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # save PCA figure
    pca_output_path = pca_output_dir / f'pca_plot_n{n_particles}_dim{n_dimensions}_steps{n_steps}_w{zone_width}_{topology}.png'
    plt.tight_layout()
    plt.savefig(pca_output_path, dpi=300, bbox_inches='tight')
    print(f"\nPCA plot saved to: {pca_output_path}")
    
    plt.close()
    
    # create inner product matrix heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(inner_products, cmap='RdBu_r', aspect='auto', interpolation='nearest')
    
    ax.set_xlabel('Particle Index', fontsize=12)
    ax.set_ylabel('Particle Index', fontsize=12)
    ax.set_title('Inner Product Matrix', fontsize=14)
    
    # add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Inner Product', fontsize=11)
    
    # save heatmap figure
    heatmap_output_path = heatmap_output_dir / f'inner_product_n{n_particles}_dim{n_dimensions}_steps{n_steps}_w{zone_width}_{topology}.png'
    plt.tight_layout()
    plt.savefig(heatmap_output_path, dpi=300, bbox_inches='tight')
    print(f"Inner product heatmap saved to: {heatmap_output_path}")
    
    plt.close()
    
    return trajectory, velocity_history, inner_products, positions_3d


def analyze_manifold_umap(n_particles: int = 100, n_dimensions: int = 6, n_steps: int = 100, zone_width: float = 5.0, topology: str = 'circle'):
    # create dynamical system with default parameters
    system = SphereDynamics(n_particles, n_dimensions, zone_width, topology)
    
    # run simulation for 100 time steps
    trajectory, velocity_history = system.simulate(n_steps=n_steps)
    
    # get final positions after simulation
    final_positions = system.positions  # shape: (n_particles, n_dimensions)
    
    # perform UMAP to reduce to 3D for visualization
    reducer = UMAP(n_components=3, random_state=42)
    positions_3d = reducer.fit_transform(final_positions)
    
    # create output directories if they don't exist
    umap_output_dir = Path("figures/umap")
    umap_output_dir.mkdir(parents=True, exist_ok=True)
    
    # create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # color points by their index (representing position along topology)
    colors = np.arange(n_particles)
    scatter = ax.scatter(
        positions_3d[:, 0], 
        positions_3d[:, 1], 
        positions_3d[:, 2],
        c=colors,
        cmap='hsv',
        s=50,
        alpha=0.7,
        edgecolors='k',
        linewidth=0.5
    )
    
    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.set_zlabel('UMAP3', fontsize=12)
    ax.set_title(f'UMAP Projection of {n_particles} Particles on {n_dimensions-1}-Sphere\n'
                 f'After {n_steps} Steps (Topology: {topology}, Zone Width: {zone_width})',
                 fontsize=14, pad=20)
    
    # add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Particle Index', fontsize=11)
    
    # save UMAP figure
    umap_output_path = umap_output_dir / f'umap_plot_n{n_particles}_dim{n_dimensions}_steps{n_steps}_w{zone_width}_{topology}.png'
    plt.tight_layout()
    plt.savefig(umap_output_path, dpi=300, bbox_inches='tight')
    print(f"\nUMAP plot saved to: {umap_output_path}")
    
    plt.close()
    
    return trajectory, velocity_history, positions_3d


def analyze_manifold_tsne(n_particles: int = 100, n_dimensions: int = 6, n_steps: int = 100, zone_width: float = 5.0, topology: str = 'circle'):
    # create dynamical system with default parameters
    system = SphereDynamics(n_particles, n_dimensions, zone_width, topology)
    
    # run simulation for 100 time steps
    trajectory, velocity_history = system.simulate(n_steps=n_steps)
    
    # get final positions after simulation
    final_positions = system.positions  # shape: (n_particles, n_dimensions)
    
    # perform t-SNE to reduce to 3D for visualization
    reducer = TSNE(n_components=3, random_state=42)
    positions_3d = reducer.fit_transform(final_positions)
    
    # create output directories if they don't exist
    tsne_output_dir = Path("figures/tsne")
    tsne_output_dir.mkdir(parents=True, exist_ok=True)
    
    # create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # color points by their index (representing position along topology)
    colors = np.arange(n_particles)
    scatter = ax.scatter(
        positions_3d[:, 0], 
        positions_3d[:, 1], 
        positions_3d[:, 2],
        c=colors,
        cmap='hsv',
        s=50,
        alpha=0.7,
        edgecolors='k',
        linewidth=0.5
    )
    
    ax.set_xlabel('t-SNE1', fontsize=12)
    ax.set_ylabel('t-SNE2', fontsize=12)
    ax.set_zlabel('t-SNE3', fontsize=12)
    ax.set_title(f't-SNE Projection of {n_particles} Particles on {n_dimensions-1}-Sphere\n'
                 f'After {n_steps} Steps (Topology: {topology}, Zone Width: {zone_width})',
                 fontsize=14, pad=20)
    
    # add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Particle Index', fontsize=11)
    
    # save t-SNE figure
    tsne_output_path = tsne_output_dir / f'tsne_plot_n{n_particles}_dim{n_dimensions}_steps{n_steps}_w{zone_width}_{topology}.png'
    plt.tight_layout()
    plt.savefig(tsne_output_path, dpi=300, bbox_inches='tight')
    print(f"\nt-SNE plot saved to: {tsne_output_path}")
    
    plt.close()
    
    return trajectory, velocity_history, positions_3d


if __name__ == "__main__":
    # analyze_manifold_pca(n_particles=100, n_dimensions=5, n_steps=100000, zone_width=5.0, topology='circle')

    # analyze_manifold_umap(n_particles=100, n_dimensions=5, n_steps=100000, zone_width=5.0, topology='circle')

    analyze_manifold_tsne(n_particles=100, n_dimensions=5, n_steps=100000, zone_width=5.0, topology='circle')