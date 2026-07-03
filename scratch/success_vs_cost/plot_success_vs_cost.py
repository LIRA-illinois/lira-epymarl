import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def create_success_vs_cost_plot() -> None:
    # load data
    load_dir = os.path.join("data")

    data = []
    comms_formulations = [
        "baseline_formulation",
        "our_formulation",
    ]
    dot_colors = [
        "darkorange",
        "black",
    ]
    labels = [
        "Baseline",
        "Ours",
    ]

    for comms_formulation in comms_formulations:
        load_path = os.path.join(load_dir, f"success_vs_cost_{comms_formulation}.csv")
        data.append(pd.read_csv(load_path))

    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    for i, df_data in enumerate(data):
        # normalize to [0, 1] by dividing by the length of the longest path in the CM
        normalized_comms_cost = df_data.total_comms_cost / 1
        ax.plot(
            df_data.success_prob_spec,
            normalized_comms_cost,
            markersize=4,
            marker="o",
            c=dot_colors[i],
            linewidth=1,
            label=labels[i],
            alpha=0.8,
        )

    ax.legend()
    ax.set_xlabel("Mission Success Prob., $p_c$", fontsize=12)
    ax.set_ylabel("Communication Cost", fontsize=12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.grid()

    plt.tight_layout()
    save_path = os.path.join(load_dir, "success_vs_cost.pdf")
    plt.savefig(save_path)
    save_path = os.path.join(load_dir, "success_vs_cost.png")
    plt.savefig(save_path, dpi=500)


create_success_vs_cost_plot()
