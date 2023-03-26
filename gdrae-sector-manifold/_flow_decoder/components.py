
import torch
import numpy as np
from torch import nn
from scipy import linalg
from functools import wraps
from torch.nn import functional as F


def api_empty_cache(func):
    @wraps(func)
    def _wrapped(*args, **kwargs):
        torch.cuda.empty_cache()
        ret = func(*args, **kwargs)
        torch.cuda.empty_cache()
        return ret

    return _wrapped


class BaseModule(nn.Module):
    """
    Base module.
    """
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def linearized_transpose(self, *args, **kwargs):
        raise NotImplementedError


########################################################################################################################
# Invertible convs activations
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Invertible convolutions
# ----------------------------------------------------------------------------------------------------------------------

class InvConv2d1x1Fixed(BaseModule):
    """
    Invconv1x1 with fixed rotation matrix as the weight.
    """
    def __init__(self, input_nc, output_nc):
        super(InvConv2d1x1Fixed, self).__init__()
        # Config.
        nc = max(input_nc, output_nc) if output_nc is not None else input_nc
        # For getting conv weight.
        if output_nc < input_nc: self._get_weight = lambda _w: _w[:output_nc].unsqueeze(-1).unsqueeze(-1)
        elif output_nc > input_nc: self._get_weight = lambda _w: _w[:, :input_nc].unsqueeze(-1).unsqueeze(-1)
        else: self._get_weight = lambda _w: _w.unsqueeze(-1).unsqueeze(-1)
        # --------------------------------------------------------------------------------------------------------------
        # Architecture
        # --------------------------------------------------------------------------------------------------------------
        self.register_buffer("_matrix_r", torch.from_numpy(linalg.qr(np.random.randn(nc, nc))[0].astype('float32')))

    # ------------------------------------------------------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        return F.conv2d(x, self._get_weight(self._matrix_r))

    def linearized_transpose(self, x):
        return F.conv_transpose2d(x, self._get_weight(self._matrix_r))


# ----------------------------------------------------------------------------------------------------------------------
# Invertible activations
# ----------------------------------------------------------------------------------------------------------------------

class InvElementwiseReLU(BaseModule):
    """
    Element-wise activation.
    """
    def __init__(self):
        super(InvElementwiseReLU, self).__init__()
        # Buffers.
        self._grads = None

    # ------------------------------------------------------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, x, linearize=False):
        # 1. Calculate output.
        output = torch.relu(x)
        # Linearize.
        if linearize: self._grads = torch.gt(x, torch.zeros_like(x)).to(x.dtype).detach()
        # Return
        return output

    @api_empty_cache
    def linearized_transpose(self, x, retain_grads=False):
        # --------------------------------------------------------------------------------------------------------------
        # Get grads.
        grads = self._grads
        """ Encountered in autograd_jacob procedure. """
        if len(x) > len(grads):
            grads = grads.unsqueeze(1).expand(grads.size(0), len(x)//len(grads), *grads.size()[1:]).reshape(*x.size())
        # --------------------------------------------------------------------------------------------------------------
        # 1. Calculate output
        output = x * grads
        """ Clear grads. """
        if not retain_grads: self._grads = None
        # Return
        return output


########################################################################################################################
# Transpose utils.
########################################################################################################################

class TransposableConv2d(BaseModule):
    """
    Transposable conv transpose layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(TransposableConv2d, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self._conv(x)

    def linearized_transpose(self, x):
        return F.conv_transpose2d(x, self._conv.weight, stride=self._conv.stride, padding=self._conv.padding)


########################################################################################################################
# Layers.
########################################################################################################################

class Coupling(BaseModule):
    """
    Coupling.
    """
    def __init__(self, input_nc, hidden_nc):
        """
        :param input_nc:
        :param hidden_nc:
        :return:
        """
        super(Coupling, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Architecture of NN.
        # --------------------------------------------------------------------------------------------------------------
        self._nn = nn.ModuleList([
            TransposableConv2d(input_nc//2, hidden_nc, kernel_size=1, stride=1, padding=0, bias=True), InvElementwiseReLU(),
            TransposableConv2d(hidden_nc, input_nc//2, kernel_size=1, stride=1, padding=0, bias=True)])

    # ------------------------------------------------------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, x, linearize=False):
        # 1. Split.
        x_a, x_b = x.chunk(2, dim=1)
        # 2. Coupling.
        # (1) NN
        output_nn = x_a
        for module in self._nn:
            kwargs = {'linearize': linearize} if isinstance(module, InvElementwiseReLU) else {}
            # Forward.
            output_nn = module(output_nn, **kwargs)
        # (2) Additive.
        output_b = x_b + output_nn
        # 3. Merge.
        output = torch.cat([x_a, output_b], dim=1)
        # Return
        return output

    def linearized_transpose(self, x, retain_grads=False):
        # 1. Split.
        x_a, x_b = x.chunk(2, dim=1)
        # 2. Coupling.
        # (1) NN.
        output_nn = x_b
        for module in self._nn[::-1]:
            kwargs = {'retain_grads': retain_grads} if isinstance(module, InvElementwiseReLU) else {}
            output_nn = module.linearized_transpose(output_nn, **kwargs)
        # (2) Additive.
        output_a = x_a + output_nn
        # 3. Merge.
        output = torch.cat([output_a, x_b], dim=1)
        # Return
        return output


class Flow(BaseModule):
    """
    Flow.
    """
    def __init__(self, input_nc, hidden_nc):
        """
        :param input_nc:
        :param hidden_nc:
        :return:
        """
        super(Flow, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Conv.
        self._conv = InvConv2d1x1Fixed(input_nc, input_nc)
        # 2. Coupling.
        self._coupling = Coupling(input_nc, hidden_nc)

    # ------------------------------------------------------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, x, linearize=False):
        # 1. Forward.
        output = self._conv(x)
        output = self._coupling(output, linearize=linearize)
        # Return
        return output

    def linearized_transpose(self, x, retain_grads=False):
        output = self._coupling.linearized_transpose(x, retain_grads=retain_grads)
        output = self._conv.linearized_transpose(output)
        # Return
        return output


class Block(BaseModule):
    """
    Block.
    """
    def __init__(self, input_nc, hidden_nc, n_flows):
        """
        :param input_nc:
        :param hidden_nc:
        :param n_flows:
        """
        super(Block, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        # Flows.
        self._flows = nn.ModuleList([])
        for index in range(n_flows):
            self._flows.append(Flow(input_nc, hidden_nc))

    def forward(self, x, linearize=False):
        for flow in self._flows:
            x = flow(x, linearize=linearize)
        # Retur
        return x

    def linearized_transpose(self, x, retain_grads=False):
        for flow in self._flows[::-1]:
            x = flow.linearized_transpose(x, retain_grads=retain_grads)
        # Return
        return x


class Decoder(BaseModule):
    """
    Decoder.
    """
    def __init__(self, nz, hidden_nc, n_flows):
        """
        For example:
            - ncs:                  256(4,4),   64(8,8),    16(16,16),  4(32,32)
            - hidden_ncs:           256,        128,        64,         32
            - middle_ns_flows:      1,          1,          1,          1
        """
        super(Decoder, self).__init__()
        ################################################################################################################
        # Architecture.
        ################################################################################################################
        self._block = Block(nz, hidden_nc, n_flows)
        self._final_conv = InvConv2d1x1Fixed(nz, nz)

    def forward(self, z, linearize=False):
        output = self._block(z.unsqueeze(-1).unsqueeze(-1), linearize=linearize)
        output = self._final_conv(output).squeeze(-1).squeeze(-1)
        # Return
        return output

    def linearized_transpose(self, x, retain_grads=False):
        output = self._final_conv.linearized_transpose(x.unsqueeze(-1).unsqueeze(-1))
        output = self._block.linearized_transpose(output).squeeze(-1).squeeze(-1)
        # Return
        return output
