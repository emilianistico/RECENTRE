import numpy as np
import matplotlib.pyplot as plt

def plot_feature_histograms(motion_data):
    """
    Plots histograms for each feature in the motion data.

    Args:
        motion_data (numpy array): Shape [N, T, D], where D is the number of features.
    """
    N, T, D = motion_data.shape
    data_flat = motion_data.reshape(N * T, D)  # Flatten into [N*T, D]

    # Create a 3x2 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    axes = axes.flatten()  # Flatten the 3x2 grid for easy indexing

    for d in range(D):
        names = ["tra_x","tra_y","tra_z","rot_x","rot_y","rot_z" ]
        units = ["mm","mm","mm","rad","rad","rad"]
        ax = axes[d]
        ax.hist(data_flat[:, d], bins=50, alpha=0.7, color="blue", edgecolor="black")
        ax.set_title(f"{names[d]} Distribution")
        ax.set_xlabel(units[d])
        ax.set_xlim((np.percentile(data_flat[:, d],1), np.percentile(data_flat[:, d].max(),75)))
        #print(np.percentile(data_flat[:, d],3))
        #print(np.percentile(data_flat[:, d].max(),75))
        ax.set_ylabel("Frequency")
        ax.grid(True)
        #tight layout
        plt.tight_layout()

    # Remove unused subplots if D < 6
    for i in range(D, 6):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()