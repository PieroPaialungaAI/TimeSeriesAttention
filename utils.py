import json
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data



def plot_attention(sample_X, attention_weights, title="Attention over Time", save_path=None):
    """
    Visualizes attention weights over the input time series.

    Args:
        sample_X: torch.Tensor or np.array of shape (seq_len, 1)
        attention_weights: torch.Tensor or np.array of shape (seq_len,)
        title: plot title
        save_path: optional path to save the figure
    """
    if isinstance(sample_X, torch.Tensor):
        sample_X = sample_X.detach().cpu().numpy().squeeze()
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy().squeeze()

    seq_len = len(sample_X)
    time_steps = np.arange(seq_len)

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.set_title(title)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Signal", color='blue')
    ax1.plot(time_steps, sample_X, color='blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Attention", color='red')
    ax2.plot(time_steps, attention_weights*10, color='red', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()