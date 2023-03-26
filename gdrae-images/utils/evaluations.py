
import numpy as np
from custom_pkg.basic.visualizers import plt


def vis_singular_values(sv, save_path):
    """
    Visualize singular values.
    :param sv: (nz, )
    :param save_path:
    :return:
    """
    # 1. Init figure.
    plt.figure(dpi=200)
    # 2. Plot.
    plt.bar(x=np.arange(len(sv)), height=sv, width=0.75, color='blue')
    plt.xlabel("latent dimension")
    plt.ylabel("SVs")
    # Save
    plt.savefig(save_path)
    plt.close()
