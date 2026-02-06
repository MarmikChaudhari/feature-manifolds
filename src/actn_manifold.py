"""
Manifold reconstruction from residual stream activations.
Extracts activations from specified layers, aggregates by character count, and plots PCA.

Config file: actn_manifold_config.yaml
"""

import re
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG_PATH = Path(__file__).parent / "actn_manifold_config.yaml"


def load_config() -> dict:
    """Load configuration from YAML file."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def parse_examples(filepath: Path) -> list[str]:
    """Parse examples from linebreak dataset file."""
    text = filepath.read_text()
    parts = re.split(r"EXAMPLE \d+\n", text)
    return [p.strip() for p in parts if p.strip()]


def load_all_data(data_dir: Path, k_values: list[int]) -> list[str]:
    """Load all examples from all k values."""
    examples = []
    for k in k_values:
        path = data_dir / f"synthetic_linebreak_k{k}.txt"
        examples.extend(parse_examples(path))
    return examples


def extract_activations(
    model, tokenizer, examples: list[str], layer: int
) -> tuple[dict[int, list[torch.Tensor]], int]:
    """
    Extract residual stream activations from specified layer, grouped by character count.
    
    Returns:
        char_to_acts: {char_count: [activation tensors]}
        max_char: maximum character count observed
    """
    char_to_acts = {}
    
    for example in tqdm(examples, desc="    Examples", leave=False):
        inputs = tokenizer(example, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Hidden states: (1, seq_len, d_model)
        hidden = outputs.hidden_states[layer][0].cpu()  # (seq_len, d_model)
        
        # Get token strings
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Track character count since last newline
        char_count = 0
        seen_newline = False
        
        for i, tok in enumerate(tokens):
            # Decode token to string
            tok_str = tokenizer.decode(inputs.input_ids[0, i:i+1])
            
            if tok_str == "\n" or tok == "<0x0A>":
                char_count = 0
                seen_newline = True
                continue
            
            # Only count positions after first newline
            if seen_newline and char_count > 0:
                if char_count not in char_to_acts:
                    char_to_acts[char_count] = []
                char_to_acts[char_count].append(hidden[i])
            
            char_count += len(tok_str)
    
    max_char = max(char_to_acts.keys()) if char_to_acts else 0
    return char_to_acts, max_char


def compute_mean_activations(char_to_acts: dict[int, list[torch.Tensor]]) -> dict[int, torch.Tensor]:
    """Compute mean activation per character count."""
    return {c: torch.stack(acts).mean(dim=0) for c, acts in char_to_acts.items()}


def run_pca(mu_dict: dict[int, torch.Tensor], n_components: int = 6):
    """Run PCA on mean activations."""
    char_values = sorted(mu_dict.keys())
    M = torch.stack([mu_dict[c] for c in char_values]).float().numpy()
    
    pca = PCA(n_components=n_components)
    M_proj = pca.fit_transform(M)
    
    return char_values, M_proj, pca


def plot_manifold(char_values: list[int], M_proj: np.ndarray, pca: PCA, save_path: Path, layer: int):
    """Create two 3D plots: first 3 PCs and next 3 PCs."""
    fig = plt.figure(figsize=(14, 6))
    
    # Color by character count
    colors = np.array(char_values)
    
    # Left: First 3 PCs
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(M_proj[:, 0], M_proj[:, 1], M_proj[:, 2], 
                      c=colors, cmap='viridis', s=20, alpha=0.8)
    ax1.plot(M_proj[:, 0], M_proj[:, 1], M_proj[:, 2], 'k-', alpha=0.3, linewidth=0.5)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax1.set_title('First 3 Principal Components')
    
    # Right: Next 3 PCs
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(M_proj[:, 3], M_proj[:, 4], M_proj[:, 5], 
                      c=colors, cmap='viridis', s=20, alpha=0.8)
    ax2.plot(M_proj[:, 3], M_proj[:, 4], M_proj[:, 5], 'k-', alpha=0.3, linewidth=0.5)
    ax2.set_xlabel(f'PC4 ({pca.explained_variance_ratio_[3]:.1%})')
    ax2.set_ylabel(f'PC5 ({pca.explained_variance_ratio_[4]:.1%})')
    ax2.set_zlabel(f'PC6 ({pca.explained_variance_ratio_[5]:.1%})')
    ax2.set_title('Next 3 Principal Components')
    
    # Colorbar
    cbar = fig.colorbar(sc1, ax=[ax1, ax2], shrink=0.6, label='Character Count')
    
    plt.suptitle(f'Character Count Manifold (Layer {layer})', fontsize=14)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def process_layer(
    model, tokenizer, examples: list[str], layer: int, fig_dir: Path
) -> None:
    """Process a single layer: extract activations, run PCA, and plot."""
    print(f"  Extracting activations...")
    char_to_acts, _ = extract_activations(model, tokenizer, examples, layer)
    
    print(f"  Computing mean activations...")
    mu_dict = compute_mean_activations(char_to_acts)
    
    # Run PCA
    char_values, M_proj, pca = run_pca(mu_dict, n_components=6)
    var_explained = sum(pca.explained_variance_ratio_[:6])
    print(f"  Variance explained by 6 PCs: {var_explained:.1%}")
    
    # Plot
    plot_manifold(char_values, M_proj, pca, fig_dir / f"manifold_pca_layer{layer}.png", layer)


def main():
    # Load config
    config = load_config()
    model_id = config['model_id']
    layers = config['layers']
    data_dir = Path(config['data_dir'])
    fig_dir = Path(config['fig_dir'])
    k_values = config['k_values']
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Config: {CONFIG_PATH}")
    print(f"Model: {model_id}")
    print(f"Layers: {layers}")
    
    # Load model once
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded on {model.device}")
    
    # Load data once
    print("\nLoading data...")
    examples = load_all_data(data_dir, k_values)
    print(f"Loaded {len(examples)} examples")
    
    # Process each layer
    for layer in layers:
        print(f"\n{'='*50}")
        print(f"Processing Layer {layer}")
        print('='*50)
        process_layer(model, tokenizer, examples, layer, fig_dir)
    
    print(f"\nDone! Processed {len(layers)} layers.")


if __name__ == "__main__":
    main()
