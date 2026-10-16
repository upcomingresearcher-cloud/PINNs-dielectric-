"""Figure: P1 FEM error vs mesh size, conforming vs non-conforming. Reads results/raw/fem_runs.csv only."""
import pathlib, sys
import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def make(root):
    d = pd.read_csv(root / "results/raw/fem_runs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, (q, lab) in zip(axes, (("primary_field_error", "interface-band field error (primary metric)"),
                                    ("relL2_phi_global", "global potential error"))):
        for k, c, mk in zip((5, 80, 2000), ("tab:blue", "tab:orange", "tab:red"), ("o", "^", "v")):
            nc = d[(d.kappa == k) & (~d.conforming)].sort_values("n_elements")
            cf = d[(d.kappa == k) & (d.conforming)].sort_values("n_elements")
            ax.loglog(nc.n_elements, nc[q], mk + "-", color=c, ms=7 if k == 80 else 5, label=f"non-conforming, κ={k}")
            ax.loglog(cf.n_elements, cf[q].clip(lower=1e-17), "s--", color=c, alpha=.5, label=f"conforming, κ={k}")
        ax.axhline(0.05, color="k", lw=1, ls=":"); ax.text(2.2, 0.06, "5% threshold", fontsize=8)
        ax.set_xlabel("number of elements N"); ax.set_ylabel(f"relative L2 error ({lab.split(' (')[0]})")
        ax.set_title(lab); ax.grid(True, which="both", alpha=.3)
    axes[0].legend(fontsize=7, ncol=1)
    fig.suptitle("P1 FEM vs mesh size. Conforming (node at interface): exact to round-off.\n"
                 "Non-conforming: band field error ~ N^-1/2, potential error ~ N^-1 (κ=80 and κ=2000 nearly coincide)", fontsize=10)
    out = root / "results/figures/fem_p1_mesh_alignment.png"; out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); return out

if __name__ == "__main__":
    print(make(pathlib.Path(__file__).resolve().parents[2]))
