import os
from os.path import join

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

font_path = join("data", "times.ttf")
import matplotlib.font_manager as fm

fm.fontManager.addfont(font_path)

# Activate the font
plt.rcParams["font.family"] = "Times New Roman"

# mpl.rc("font", family="Times")


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
    ax.set_xlabel("Task Satis. Rate", fontsize=13)
    # ax.set_xlabel("Mission Success Prob., $p_c$", fontsize=12)
    ax.set_ylabel("Network Cost", fontsize=13)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.grid()

    plt.tight_layout()

    # place the arrow
    text_x = 0.95
    plt.annotate(
        "",
        xy=(0.975, 0.52),  # (x, y) coordinates where the arrow points
        xytext=(0.975, 0.73),  # (x, y) coordinates where the text is placed
        arrowprops=dict(color="red"),  # Makes the arrow red
    )
    plt.text(x=text_x, y=0.68, s="Lower\ncost", color="red")

    # place the label separately
    # plt.annotate(
    #     "Lower cost",  # The text label
    #     color="red",
    #     xy=(0.98, 0.5),  # (x, y) coordinates where the arrow points
    #     xytext=(0.98, 0.75),  # (x, y) coordinates where the text is placed
    #     arrowprops=dict(facecolor="red"),  # Makes the arrow red
    # )

    # save_path = os.path.join(load_dir, "success_vs_cost.pdf")
    # plt.savefig(save_path)
    save_path = os.path.join(load_dir, "success_vs_cost.png")
    # plt.show()
    plt.savefig(save_path, dpi=500)
    plt.close()


create_success_vs_cost_plot()
