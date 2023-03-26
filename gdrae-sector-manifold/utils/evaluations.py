
import math
from custom_pkg.basic.visualizers import plt, gradient_colors


def _vis_grids(data2d):
    """
    :param data2d: (n_grids, n_grids, 2)
    :return:
    """
    # 1. Along axis 0.
    colors = gradient_colors(data2d.shape[0], change='blue2green')
    for row_index in range(data2d.shape[0]):
        # Plot current row_line
        for col_index in range(1, data2d.shape[1]):
            pairs = data2d[row_index, col_index-1:col_index+1]
            plt.plot(pairs[:, 0], pairs[:, 1], color=colors[row_index])
    # 2. Along axis 1.
    colors = gradient_colors(data2d.shape[1], change='blue2red')
    for col_index in range(data2d.shape[1]):
        # Plot current col_line
        for row_index in range(1, data2d.shape[0]):
            pairs = data2d[row_index-1:row_index+1, col_index]
            plt.plot(pairs[:, 0], pairs[:, 1], color=colors[col_index])


def vis_correspondence(latent, manifold, save_path):
    """
    :param latent: (n_grids**2, 2)
    :param manifold: (n_grids**2, 2)
    :param save_path:
    :return:
    """
    n_grids = int(math.sqrt(len(latent)))
    assert n_grids**2 == len(latent)
    # 1. Init figure.
    plt.figure(dpi=200)
    # 2. Figures.
    # (1) Latent.
    plt.subplot(121, aspect=1.0)
    plt.title("Latent")
    _vis_grids(latent.reshape(n_grids, n_grids, 2))
    # (2) Manifold.
    plt.subplot(122, aspect=1.0)
    plt.title("Manifold")
    _vis_grids(manifold.reshape(n_grids, n_grids, 2))
    # Save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def vis_rec_gt_manifold(x_gt, x_recon, save_path):
    """
    :param x_gt: (n, 2)
    :param x_recon: (n, 2)
    :param save_path:
    :return:
    """
    # 1. Init figure.
    plt.figure(dpi=200)
    # 2. Figures.
    # (1) Real manifold.
    plt.subplot(121, aspect=1.0)
    plt.title("Real manifold")
    plt.scatter(x=x_gt[:, 0], y=x_gt[:, 1], s=0.5)
    # (2) Reconstruction manifold.
    plt.subplot(122, aspect=1.0)
    plt.title("Reconstructed manifold")
    plt.scatter(x=x_recon[:, 0], y=x_recon[:, 1], s=0.5)
    # Save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def vis_mu_scatter(mu, save_path):
    # 1. Init figure.
    plt.figure(dpi=200)
    # 2. Figures.
    plt.title("Scatter of " + r"$\mu$")
    plt.scatter(x=mu[:, 0], y=mu[:, 1], s=0.5)
    plt.tight_layout()
    # Save
    plt.savefig(save_path)
    plt.close()
