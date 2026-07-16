"""Publication-quality figure for the ADAPT router simulation campaign.

Reads the most recent timestamped simulation-batch CSV produced by
tests/test_simulation_batch.py and renders a two-panel figure:

  - Left : recovered m1 vs m2 scatter, colored by routing decision, with
           the neutron-star (2.2 Msun) and black-hole (5.0 Msun) mass
           boundaries drawn in.
  - Right: match / ambiguous / mismatch bucket counts with an accuracy
           and safe-path-rate summary box.

Output PDFs are timestamped so each run produces a new file rather than
overwriting the previous figure. PDF is used for publication-quality
vector graphics (scales cleanly in manuscripts).
"""

import glob
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ROUTE_COLORS = {"BNS": "#1f77b4", "BBH": "#d62728", "AMBIGUOUS": "#ff7f0e"}
ROUTE_LABELS = {"BNS": "Routed: BNS", "BBH": "Routed: BBH", "AMBIGUOUS": "Routed: AMBIGUOUS"}


def find_latest_simulation_csv() -> str:
    """Return the newest simulation_batch_*.csv (falling back to the legacy name)."""
    candidates = glob.glob(os.path.join(RESULTS_DIR, "simulation_batch_*.csv"))
    legacy = os.path.join(RESULTS_DIR, "simulation_batch.csv")
    if os.path.exists(legacy):
        candidates.append(legacy)
    if not candidates:
        raise FileNotFoundError(
            f"No simulation results found in {RESULTS_DIR}. "
            "Run 'python tests/test_simulation_batch.py' first!"
        )
    return max(candidates, key=os.path.getmtime)


def generate_academic_plot(csv_path: str = None) -> str:
    if csv_path is None:
        csv_path = find_latest_simulation_csv()
    print(f"Reading simulation results from: {csv_path}")

    df = pd.read_csv(csv_path)

    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]})

    # ---------------- LEFT PANEL: physical mass space ----------------
    for route_type, group in df.groupby("route"):
        ax1.scatter(
            group["recovered_m1"],
            group["recovered_m2"],
            color=ROUTE_COLORS.get(route_type, "#333333"),
            label=ROUTE_LABELS.get(route_type, f"Routed: {route_type}"),
            alpha=0.6,
            edgecolors="none",
            s=25,
        )

    ax1.set_title("ADAPT Router Physical Classification Space", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel(r"Recovered Primary Mass $m_1$ ($M_\odot$)", fontsize=12)
    ax1.set_ylabel(r"Recovered Secondary Mass $m_2$ ($M_\odot$)", fontsize=12)
    ax1.set_xlim(0, 55)
    ax1.set_ylim(0, 55)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # NS ceiling (2.2) and BH floor (5.0) boundary lines.
    for boundary in (2.2, 5.0):
        ax1.axvline(boundary, color="gray", linestyle="--", alpha=0.5)
        ax1.axhline(boundary, color="gray", linestyle="--", alpha=0.5)

    # ---------------- RIGHT PANEL: performance metrics ----------------
    bucket_counts = df["bucket"].value_counts().to_dict()
    for category in ("match", "ambiguous", "mismatch"):
        bucket_counts.setdefault(category, 0)

    categories = ["Exact Matches", "Safe Ambiguous", "Catastrophic Mismatches"]
    values = [bucket_counts["match"], bucket_counts["ambiguous"], bucket_counts["mismatch"]]
    bar_colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    bars = ax2.bar(categories, values, color=bar_colors, edgecolor="black", width=0.5)
    ax2.set_title("Routing Accuracy Breakdown", fontsize=14, fontweight="bold", pad=15)
    ax2.set_ylabel("Count (Event Triggers)", fontsize=12)

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    total_samples = len(df)
    accuracy = 100.0 * bucket_counts["match"] / total_samples
    safe_rate = 100.0 * (bucket_counts["match"] + bucket_counts["ambiguous"]) / total_samples
    mismatch_rate = 100.0 * bucket_counts["mismatch"] / total_samples

    stats_text = (
        f"Total Samples: {total_samples}\n"
        f"Exact Match Accuracy: {accuracy:.2f}%\n"
        f"Safe Path Rate: {safe_rate:.2f}%\n"
        f"Catastrophic Mismatch: {mismatch_rate:.2f}%"
    )
    ax2.text(
        0.5,
        0.5,
        stats_text,
        transform=ax2.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="wheat", alpha=0.2),
        ha="center",
        va="center",
    )

    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_pdf_path = os.path.join(RESULTS_DIR, f"router_performance_{timestamp}.pdf")
    plt.savefig(save_pdf_path, format="pdf", bbox_inches="tight")
    print(f"\nAcademic figure successfully created and saved to: {save_pdf_path}")
    return save_pdf_path


if __name__ == "__main__":
    generate_academic_plot()
