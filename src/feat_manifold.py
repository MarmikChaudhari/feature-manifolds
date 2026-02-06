"""
Feature manifold from SAE/transcoder/CLT reconstruction + tuning-curve analysis.

Extracts residual-stream activations, encodes through SAE (sae-lens) or
Cross-Layer Transcoder (CLT, loaded directly from Gemma Scope 2 weights),
selects character-count-tuned features, reconstructs the manifold from those
features, and plots:
  1. Reconstructed manifold in PCA space (same basis as activation manifold)
  2. Feature tuning curves + active-feature count

Caches activations, PCA basis, and feature results as .pt files.
Config: feat_manifold_config.yaml
"""

import json
import re
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors import safe_open

CONFIG_PATH = Path(__file__).parent / "feat_manifold_config.yaml"


# ── Cross-Layer Transcoder ──────────────────────────────────────


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """Gemma-style RMSNorm: x / RMS(x) * weight."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight


class GemmaScope2CLT:
    """Cross-Layer Transcoder loader for Gemma Scope 2 HuggingFace weights.

    Downloads per-layer encoder/decoder weights and provides encode()/decode()
    compatible with the feature manifold pipeline.

    Weight layout per layer file (params_layer_{L}.safetensors):
        w_enc:      [d_model, d_feat]
        b_enc:      [d_feat]
        threshold:  [d_feat]            (JumpReLU gate)
        w_dec:      [d_feat, n_layers, d_model]  (cross-layer write)
        b_dec:      [d_model]
    """

    def __init__(
        self,
        repo_id: str,
        clt_id: str,
        layer: int,
        device: str,
        ln_weight: torch.Tensor | None = None,
    ):
        path = hf_hub_download(
            repo_id, f"clt/{clt_id}/params_layer_{layer}.safetensors",
        )
        with safe_open(path, framework="pt") as f:
            self.W_enc = f.get_tensor("w_enc").to(device)          # [d_model, d_feat]
            self.b_enc = f.get_tensor("b_enc").to(device)          # [d_feat]
            self.threshold = f.get_tensor("threshold").to(device)  # [d_feat]
            self.W_dec = f.get_tensor("w_dec").to(device)          # [d_feat, n_layers, d_model]
            self.b_dec = f.get_tensor("b_dec").to(device)          # [d_model]

        # Optional pre-feedforward layernorm weight for RMSNorm
        self.ln_weight = ln_weight.to(device) if ln_weight is not None else None

        self.layer = layer
        self.device = device
        self.d_in = self.W_enc.shape[0]
        self.d_sae = self.W_enc.shape[1]   # per-layer feature count
        self.n_layers = self.W_dec.shape[1]

        # Load CLT config for metadata
        cfg_path = hf_hub_download(repo_id, f"clt/{clt_id}/config.json")
        with open(cfg_path) as f_cfg:
            self.config = json.load(f_cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """JumpReLU encoding with optional RMSNorm pre-processing."""
        x = x.to(self.device)
        if self.ln_weight is not None:
            x = rms_norm(x, self.ln_weight)
        pre_act = x @ self.W_enc + self.b_enc
        return torch.where(
            pre_act > self.threshold, pre_act, torch.zeros_like(pre_act),
        )

    def decode(
        self, features: torch.Tensor, output_layer: int | None = None,
    ) -> torch.Tensor:
        """Decode features to a specific output layer (defaults to encoder layer)."""
        features = features.to(self.device)
        if output_layer is None:
            output_layer = self.layer
        return features @ self.W_dec[:, output_layer, :] + self.b_dec


# ── Data loading ────────────────────────────────────────────────────


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def parse_examples(path: Path) -> list[str]:
    parts = re.split(r"EXAMPLE \d+\n", path.read_text())
    return [p.strip() for p in parts if p.strip()]


def load_data(data_dir: Path, k_values: list[int]) -> list[str]:
    out = []
    for k in k_values:
        out.extend(parse_examples(data_dir / f"synthetic_linebreak_k{k}.txt"))
    return out


# ── Activation extraction + caching ────────────────────────────────


def extract_activations(
    model, tokenizer, examples: list[str], hs_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual stream at hidden_states[hs_index], labelled by line char count.

    Returns (acts [N, D], char_counts [N]).
    """
    acts, ccs = [], []
    for ex in tqdm(examples, desc="Extracting activations"):
        inp = tokenizer(ex, return_tensors="pt").to(model.device)
        with torch.no_grad():
            hs = model(
                **inp, output_hidden_states=True
            ).hidden_states[hs_index][0].cpu().float()
        cc, seen = 0, False
        for i in range(inp.input_ids.shape[1]):
            s = tokenizer.decode(inp.input_ids[0, i : i + 1])
            tok = tokenizer.convert_ids_to_tokens(inp.input_ids[0, i : i + 1])[0]
            if s == "\n" or tok == "<0x0A>":
                cc, seen = 0, True
                continue
            if seen and cc > 0:
                acts.append(hs[i])
                ccs.append(cc)
            cc += len(s)
    return torch.stack(acts), torch.tensor(ccs)


def cached_activations(model, tokenizer, examples, hs_index, cache_dir, layer):
    """Load cached activations or extract and save."""
    path = cache_dir / f"activations_layer{layer}.pt"
    if path.exists():
        print(f"  Cache hit: {path}")
        d = torch.load(path, weights_only=True)
        return d["acts"], d["ccs"]
    acts, ccs = extract_activations(model, tokenizer, examples, hs_index)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"acts": acts, "ccs": ccs}, path)
    print(f"  Cached: {path}")
    return acts, ccs


# ── PCA from activation manifold centroids ──────────────────────────


def build_pca(
    acts: torch.Tensor,
    ccs: torch.Tensor,
    n_comp: int,
    cache_dir: Path,
    layer: int,
) -> tuple[list[int], np.ndarray, PCA]:
    """Fit PCA on per-char-count mean activations.

    Returns (sorted_char_values, mu_bar, fitted_pca).
    """
    path = cache_dir / f"pca_layer{layer}.pt"
    if path.exists():
        print(f"  Cache hit: {path}")
        d = torch.load(path, weights_only=True)
        pca = PCA(n_components=n_comp)
        pca.components_ = d["P"].numpy()
        pca.mean_ = d["pca_mean"].numpy()
        pca.explained_variance_ratio_ = d["evr"].numpy()
        return d["cv"].tolist(), d["mu_bar"].numpy(), pca

    cv = sorted(ccs.unique().tolist())
    M = torch.stack([acts[ccs == c].mean(0) for c in cv]).numpy()
    pca = PCA(n_components=n_comp).fit(M)
    mu_bar = M.mean(axis=0)

    torch.save(
        {
            "cv": torch.tensor(cv),
            "mu_bar": torch.from_numpy(mu_bar),
            "P": torch.from_numpy(pca.components_.copy()),
            "pca_mean": torch.from_numpy(pca.mean_.copy()),
            "evr": torch.from_numpy(pca.explained_variance_ratio_.copy()),
        },
        path,
    )
    print(f"  Cached: {path}")
    return cv, mu_bar, pca


# ── Feature activations ───────────────────────────────────────


def mean_feature_acts(
    encoder,
    acts: torch.Tensor,
    ccs: torch.Tensor,
    device: str,
    batch_size: int = 512,
) -> dict[int, torch.Tensor]:
    """Encode all activations through encoder, return mean feature acts per char count."""
    chunks = []
    for i in tqdm(range(0, len(acts), batch_size), desc="Encode"):
        with torch.no_grad():
            chunks.append(encoder.encode(acts[i : i + batch_size].to(device)).cpu())
    feats = torch.cat(chunks)  # (N, d_sae)
    cv = sorted(ccs.unique().tolist())
    return {c: feats[ccs == c].mean(0) for c in cv}


# ── Feature selection ───────────────────────────────────────────────


def select_features(
    mf: dict[int, torch.Tensor],
    cv: list[int],
    n: int,
    method: str = "tuning",
) -> tuple[list[int], list[int]]:
    """Select top-n features by character-count tuning.

    Returns (sorted feature indices, their peak char counts).
    """
    M = torch.stack([mf[c] for c in cv])  # (C, d_sae)
    peak = M.max(dim=0).values
    if method == "tuning":
        score = peak / (M.mean(dim=0) + 1e-10)
    elif method == "variance":
        score = M.var(dim=0)
    else:
        raise ValueError(f"Unknown method: {method}")
    score[peak < 1e-6] = 0  # ignore dead features

    idx = score.topk(n).indices.tolist()
    # Sort by peak char count (left→right on x-axis)
    peaks = {i: cv[M[:, i].argmax().item()] for i in idx}
    idx.sort(key=lambda i: peaks[i])
    return idx, [peaks[i] for i in idx]


# ── Reconstruction + projection ─────────────────────────────────────


def reconstruct_project(
    mf: dict[int, torch.Tensor],
    selected: list[int],
    encoder,
    cv: list[int],
    mu_bar: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    """Reconstruct mean activations from selected features, project via PCA.

    Reconstruction: x̂_c = decode(masked ā(c))   where only selected features are kept.
    Projection:     ẑ_c = (x̂_c − μ̄_act) @ Pᵀ   using activation manifold centering.
    """
    recons = []
    for c in cv:
        f = torch.zeros_like(mf[c])
        f[selected] = mf[c][selected]
        with torch.no_grad():
            recons.append(
                encoder.decode(f.unsqueeze(0).to(encoder.device))
                .squeeze(0).cpu().float()
            )
    R = torch.stack(recons).numpy()
    return (R - mu_bar) @ pca.components_.T


# ── Plots ───────────────────────────────────────────────────────────

FEAT_COLORS = [
    "#e6194b",  # red
    "#f58231",  # orange
    "#ffe119",  # yellow
    "#bfef45",  # lime
    "#3cb44b",  # green
    "#42d4f4",  # cyan
    "#4363d8",  # blue
    "#911eb4",  # purple
    "#f032e6",  # magenta
    "#000000",  # black
    "#a9a9a9",  # gray
    "#800000",  # maroon
    "#9A6324",  # brown
    "#469990",  # teal
]


def plot_manifold(
    cv: list[int],
    Rp: np.ndarray,
    pca: PCA,
    save_path: Path,
    layer: int,
    peaks: list[int],
):
    """3D PCA of reconstructed feature manifold."""
    fig = plt.figure(figsize=(14, 6))
    c = np.array(cv)
    for sp, pcs, title in [(121, (0, 1, 2), "PCs 1–3"), (122, (3, 4, 5), "PCs 4–6")]:
        ax = fig.add_subplot(sp, projection="3d")
        ax.scatter(
            Rp[:, pcs[0]], Rp[:, pcs[1]], Rp[:, pcs[2]],
            c=c, cmap="viridis", s=20, alpha=0.8,
        )
        ax.plot(
            Rp[:, pcs[0]], Rp[:, pcs[1]], Rp[:, pcs[2]],
            "k-", alpha=0.3, lw=0.5,
        )
        # Decoder markers at peak char count positions
        for pk in peaks:
            if pk in cv:
                j = cv.index(pk)
                ax.scatter(
                    Rp[j, pcs[0]], Rp[j, pcs[1]], Rp[j, pcs[2]],
                    marker="x", s=120, c="red", linewidths=2.5, zorder=10,
                )
        evr = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC{pcs[0]+1} ({evr[pcs[0]]:.1%})")
        ax.set_ylabel(f"PC{pcs[1]+1} ({evr[pcs[1]]:.1%})")
        ax.set_zlabel(f"PC{pcs[2]+1} ({evr[pcs[2]]:.1%})")
        ax.set_title(f"Feature Manifold: {title}")
    fig.colorbar(
        fig.axes[0].collections[0], ax=fig.axes, shrink=0.6, label="Char Count",
    )
    plt.suptitle(f"Reconstructed Feature Manifold (Layer {layer})", fontsize=14)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_tuning(
    cv: list[int],
    mf: dict[int, torch.Tensor],
    selected: list[int],
    save_path: Path,
    threshold: float,
):
    """Stacked tuning curves + active feature count (two-panel figure)."""
    M = torch.stack([mf[c] for c in cv])[:, selected].numpy()  # (C, K)
    mx = M.max(axis=0, keepdims=True)
    mx[mx == 0] = 1
    normed = M / mx

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.08},
    )

    K = len(selected)
    for k in range(K):
        col = FEAT_COLORS[k % len(FEAT_COLORS)]
        ax1.plot(cv, normed[:, k], color=col, lw=1.5, label=f"LCC{k}")
        ax1.fill_between(cv, 0, normed[:, k], alpha=0.12, color=col)
    ax1.set_ylabel(r"$\bar{a}_k(c)\;/\;\max_c \bar{a}_k(c)$")
    ax1.set_title("Line Character Count vs Average Feature Activation")
    ax1.legend(loc="upper right", ncol=min(K, 5), fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    # Active feature count
    n_active = (normed > threshold).sum(axis=1)
    ax2.bar(cv, n_active, width=1.0, color="steelblue", alpha=0.7, edgecolor="none")
    ax2.set_xlabel("Line Character Count")
    ax2.set_ylabel("# Active")
    ax2.set_title(f"Active features (τ = {threshold})", fontsize=10)
    ax2.axhline(2, color="gray", ls="--", alpha=0.5, label="n=2")
    ax2.axhline(3, color="gray", ls=":", alpha=0.5, label="n=3")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ────────────────────────────────────────────────────────────


def main():
    cfg = load_config()
    layer = cfg["sae_layer"]
    hs_index = layer + 1  # hidden_states: [emb, block_0, block_1, ..., block_N]
    data_dir = Path(cfg["data_dir"])
    fig_dir = Path(cfg["fig_dir"])
    cache_dir = Path(cfg["cache_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sae_type = cfg.get("sae_type", "transcoder")

    if sae_type == "clt":
        encoder_label = f"CLT {cfg['clt_id']}  (layer {layer}, repo {cfg['clt_repo']})"
        encoder_id = f"clt_{cfg['clt_id']}"
    else:
        encoder_label = f"{cfg['sae_release']} / {cfg['sae_id']}"
        encoder_id = cfg["sae_id"]

    print(f"Model   : {cfg['model_id']}")
    print(f"Layer   : {layer}  (hidden_states[{hs_index}])")
    print(f"Encoder : {encoder_label}")

    # ── Model + data ──
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=torch.bfloat16, device_map="auto",
    )
    device = str(next(model.parameters()).device)
    examples = load_data(data_dir, cfg["k_values"])
    print(f"  {len(examples)} examples")

    # Save layernorm weight for CLT normalization (before freeing model)
    ln_weight = None
    if sae_type == "clt":
        ln_weight = (
            model.model.layers[layer].pre_feedforward_layernorm.weight.cpu().float()
        )
        print(f"  Saved pre_feedforward_layernorm weight (d={ln_weight.shape[0]})")

    # ── Activations (cached) ──
    print("\nActivations...")
    acts, ccs = cached_activations(
        model, tokenizer, examples, hs_index, cache_dir, layer,
    )
    print(f"  {acts.shape[0]} tokens, d_model={acts.shape[1]}")

    # ── PCA basis from activation manifold ──
    print("\nPCA (activation manifold centroids)...")
    cv, mu_bar, pca = build_pca(acts, ccs, cfg["n_pca_components"], cache_dir, layer)
    evr_sum = sum(pca.explained_variance_ratio_[: cfg["n_pca_components"]])
    print(f"  {len(cv)} char values, {cfg['n_pca_components']}-PC variance={evr_sum:.1%}")

    # ── Free model, load encoder ──
    del model
    torch.cuda.empty_cache()

    print(f"\nLoading {sae_type.upper()}...")
    if sae_type == "clt":
        encoder = GemmaScope2CLT(
            cfg["clt_repo"], cfg["clt_id"], layer, device, ln_weight=ln_weight,
        )
        d_sae = encoder.d_sae
    else:
        from sae_lens import SAE

        sae, cfg_dict, _ = SAE.from_pretrained(
            release=cfg["sae_release"], sae_id=cfg["sae_id"],
        )
        encoder = sae.to(device)
        d_sae = encoder.W_dec.shape[0]
    print(f"  d_sae={d_sae}")

    # ── Feature activations ──
    print("\nFeature activations...")
    mf = mean_feature_acts(encoder, acts, ccs, device)

    # ── Feature selection ──
    selected, peaks = select_features(
        mf, cv, cfg["n_features"], cfg["selection_method"],
    )
    print(f"  Selected features: {selected}")
    print(f"  Peak char counts:  {peaks}")

    # ── Save feature results ──
    results_path = cache_dir / f"feat_{encoder_id}.pt"
    torch.save(
        {
            "selected": selected,
            "peaks": peaks,
            "mean_feats": {c: mf[c] for c in cv},
            "char_values": cv,
            "sae_type": sae_type,
            "encoder_id": encoder_id,
        },
        results_path,
    )
    print(f"  Saved: {results_path}")

    # ── Reconstruct + project ──
    print("\nReconstructing manifold...")
    Rp = reconstruct_project(mf, selected, encoder, cv, mu_bar, pca)

    # Save projected manifold for quick re-plotting
    torch.save(
        {"Rp": torch.from_numpy(Rp), "cv": cv, "peaks": peaks, "selected": selected},
        cache_dir / f"feat_proj_{encoder_id}.pt",
    )

    # ── Plots ──
    print("\nPlotting...")
    plot_manifold(
        cv, Rp, pca,
        fig_dir / f"feat_manifold_layer{layer}.png",
        layer, peaks,
    )
    plot_tuning(
        cv, mf, selected,
        fig_dir / f"tuning_curves_layer{layer}.png",
        cfg["active_threshold"],
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
