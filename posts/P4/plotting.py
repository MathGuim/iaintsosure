import pandas as pd
import matplotlib.pyplot as plt
import numpy.typing as npt
from matplotlib.legend_handler import HandlerPathCollection


def plot_selected_outliers(signals: npt.NDArray, scores: npt.NDArray, contamination: float):
    plt.figure(figsize=(20, 12))
    plt.title("Selected Outliers");

    num_outliers = int(contamination * 100)
    sorted_scores_idx = scores.argsort()[:num_outliers]

    for i in range(num_outliers):
        plt.subplot(num_outliers, 1, i + 1)
        plt.plot(signals[sorted_scores_idx][i])


def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])


def plot_outlier_scores(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.scatter(df["PC1"], df["PC2"], color="k", s=3.0, label="Data points")
    # plot circles with radius proportional to the outlier scores
    radius = (df["score"].max() - df["score"]) / (df["score"].max() - df["score"].min())
    scatter = plt.scatter(
        df["PC1"],
        df["PC2"],
        s=1000 * radius,
        edgecolors="r",
        facecolors="none",
        label="Outlier scores",
    )
    plt.axis("tight")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(
        handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
    )
    plt.suptitle("Local Outlier Factor (LOF) Scores calculated with FFT Feature Representation");