
import torch
from functools import wraps
from torch.autograd import grad


def api_empty_cache(func):
    @wraps(func)
    def _wrapped(*args, **kwargs):
        torch.cuda.empty_cache()
        ret = func(*args, **kwargs)
        torch.cuda.empty_cache()
        return ret

    return _wrapped


########################################################################################################################
# Autograd
########################################################################################################################

def autograd_proc(eps, ipt, opt, create_graph=False):
    """
    :param eps: (n, output_c, output_h, output_w)
    :param ipt: (n, input_c, input_h, input_w)
    :param opt: (n, output_c, output_h, output_w)
    :param create_graph:
    :return: A vector computed as J^\top(ipt)*eps, which is in the same shape as ipt.
    """
    y = (opt * eps).sum()
    grads = grad(outputs=y, inputs=ipt, grad_outputs=torch.ones_like(y), create_graph=create_graph, retain_graph=True)[0]
    # Return
    return grads if create_graph else grads.detach()


def _batch_diag(x):
    """
    :param x: (n, dim, dim)
    :return: (n, dim)
    """
    n = x.size(1)
    indices = torch.arange(n**2, dtype=torch.int64, device=x.device).reshape(n, n)
    indices = torch.diag(indices)
    return torch.index_select(x.reshape(x.size(0), -1), dim=1, index=indices)


@api_empty_cache
def autograd_jacob(x, func, bsize, x_clip=None, y_clip=None):
    """
    :param x: (n, ...)
    :param func:
    :param bsize:
    :param x_clip: indices.
    :param y_clip: indices.
    :return: The Jacobian matrix. (n, ny, nx)
    """
    ####################################################################################################################
    # 1. Get x (n, bsize, ...) & y (n, bsize, ny')
    ####################################################################################################################
    x = x.unsqueeze(dim=1).expand(x.size(0), bsize, *x.size()[1:]).requires_grad_(True)
    y = func(x.reshape(-1, *x.size()[2:])).reshape(x.size(0), bsize, -1)
    if y_clip is not None: y = y[:, :, y_clip]
    ####################################################################################################################
    # 2. Calculate.
    ####################################################################################################################
    # (1) Init results.
    results, counts, max_counts = [], 0, y.size(2)
    # (2) Each batch.
    while counts < max_counts:
        # 1. Select outputs.
        batch_size = min(bsize, max_counts-counts)
        outputs = _batch_diag(y[:, 0:batch_size, counts:counts+batch_size])
        # 2. Calculate grads. (n, bsize, ...) -> (n, batch_size, -1)
        grads = grad(outputs=outputs, inputs=x, grad_outputs=torch.ones_like(outputs), create_graph=False, retain_graph=True)[0]
        grads = grads[:, 0:batch_size].reshape(grads.size(0), batch_size, -1)
        if x_clip is not None: grads = grads[:, :, x_clip]
        # Save
        results.append(grads.detach())
        # Move forward
        counts += batch_size
    ####################################################################################################################
    # 3. Get Jacobian matrix. (n, ny, nx)
    return torch.cat(results, dim=1)


########################################################################################################################
# Calculations.
########################################################################################################################

def l2_normalization(x):
    sizes = x.size()[1:]
    # 1. Reshape.
    x = x.reshape(x.size(0), -1)
    # 2. Normalize.
    x = x / ((x**2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
    # 3. Reshape.
    x = x.reshape(x.size(0), *sizes)
    # Return
    return x


def normalized_mean_absolute_err(pred, gt):
    """
    NMAE.
    :param pred:
    :param gt:
    :return:
    """
    # 1. Calculate Mean Absolute Err (MAE).
    mae = (pred - gt).abs().mean()
    # 2. Normalize.
    norm = gt.abs().mean()
    # Return
    return (mae / (norm+1e-8)).item()


def measure_orthogonality(matrix):
    """
    :param matrix: (batch, n, n)
    """
    indicator = torch.ones_like(matrix) == torch.eye(matrix.size(1), dtype=matrix.dtype, device=matrix.device).unsqueeze(0)
    elem_diag, elem_other = map(lambda _i: matrix[_i].reshape(matrix.size(0), -1), [indicator, ~indicator])
    # 1. Get orthogonality significance = magnitude(other_locations) / magnitude(diagonals). (batch, )
    mag_diag = elem_diag.abs().mean(dim=1)
    mag_other = elem_other.abs().mean(dim=1)
    ortho_sign = mag_other / (mag_diag + 1e-8)
    # Return
    return ortho_sign, elem_diag
