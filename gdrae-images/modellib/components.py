
import math
import numpy as np
from torch import nn
from scipy import linalg
from utils.operations import *
from torch.nn import functional as F


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
    def __init__(self, input_nc, output_nc, **kwargs):
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
        if 'matrix_r' in kwargs.keys():
            matrix_r = kwargs['matrix_r']
            assert len(matrix_r.size()) == 2 and matrix_r.size(0) == matrix_r.size(1) == nc
        else: matrix_r = torch.from_numpy(linalg.qr(np.random.randn(nc, nc))[0].astype('float32'))
        """ Set weight. """
        self.register_buffer("_matrix_r", matrix_r)

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

class InvElementWiseAct(BaseModule):
    """
    Element-wise activation.
    """
    def __init__(self):
        super(InvElementWiseAct, self).__init__()
        # Buffers.
        self._grads = None

    def forward(self, x, linearize=False):
        raise NotImplementedError

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


class InvElementwiseReLU(InvElementWiseAct):
    """
    Element-wise activation.
    """
    def forward(self, x, linearize=False):
        # 1. Calculate output.
        output = torch.relu(x)
        # Linearize.
        if linearize: self._grads = torch.gt(x, torch.zeros_like(x)).to(x.dtype).detach()
        # Return
        return output


class InvElementwiseSigmoid(InvElementWiseAct):
    """
    Element-wise activation.
    """
    def forward(self, x, linearize=False):
        # 1. Calculate output.
        output = torch.sigmoid(x)
        # Linearize.
        if linearize: self._grads = (output * (1.0 - output)).detach()
        # Return
        return output


########################################################################################################################
# Transpose utils.
########################################################################################################################

class Squeeze(BaseModule):
    """
    Squeeze Fn.
    """
    def __init__(self, s=2):
        super(Squeeze, self).__init__()
        # Config.
        self._s = s

    def forward(self, x):
        return squeeze_nc(x, s=self._s)

    def linearized_transpose(self, x):
        return unsqueeze_nc(x, s=self._s)


class Unsqueeze(BaseModule):
    """
    Unsqueeze Fn.
    """
    def __init__(self, s=2):
        super(Unsqueeze, self).__init__()
        # Config.
        self._s = s

    def forward(self, x):
        return unsqueeze_nc(x, s=self._s)

    def linearized_transpose(self, x):
        return squeeze_nc(x, s=self._s)


class ConcatZero(BaseModule):
    """
    Concat zero Fn.
    """
    def forward(self, x):
        return torch.cat([x, torch.zeros_like(x)], dim=1)

    def linearized_transpose(self, x):
        return x.chunk(2, dim=1)[0]


class TransposableConvTranspose2d(BaseModule):
    """
    Transposable conv transpose layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(TransposableConvTranspose2d, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        self._convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self._convt(x)

    def linearized_transpose(self, x):
        return F.conv2d(x, self._convt.weight, stride=self._convt.stride, padding=self._convt.padding)


########################################################################################################################
# Layers.
########################################################################################################################

class DimIncreaseUnsqueeze2(BaseModule):
    """
    Use a invertible conv to increase dim first, then unsqueeze feature map.
    """
    def __init__(self, input_nc, output_nc):
        super(DimIncreaseUnsqueeze2, self).__init__()
        # Config.
        assert output_nc*4 >= input_nc
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        self._conv = InvConv2d1x1Fixed(input_nc, output_nc*4)
        self._usqz = Unsqueeze()

    def forward(self, x):
        output = self._conv(x)
        output = self._usqz(output)
        # Return
        return output

    def linearized_transpose(self, x):
        output = self._usqz.linearized_transpose(x)
        output = self._conv.linearized_transpose(output)
        # Return
        return output


class Coupling(BaseModule):
    """
    Coupling.
    """
    def __init__(self, input_nc, input_size, hidden_nc1, hidden_nc2, **kwargs):
        """
        :param input_nc:
        :param input_size:
        :param hidden_nc1:
        :param hidden_nc2:
        :param kwargs:
            - ds_conv: For init layer.
        :return:
        """
        super(Coupling, self).__init__()
        # Config.
        init_phase = True if 'ds_conv' in kwargs.keys() else False
        # --------------------------------------------------------------------------------------------------------------
        # Architecture of NN.
        # --------------------------------------------------------------------------------------------------------------
        self._nn = nn.ModuleList([])
        # 1. Downsampling. (input_nc//2, input_size, input_size) -> (hidden_nc1, input_size//2, input_size//2)
        self._nn.append(Squeeze(s=input_size if init_phase else 2))
        self._nn.append(kwargs['ds_conv'] if init_phase else InvConv2d1x1Fixed(input_nc*2, hidden_nc1))
        # 2. Hidden. (hidden_nc1, input_size//2, input_size//2) -> (hidden_nc2, input_size, input_size)
        convt_kwargs = {'kernel_size': input_size, 'stride': 1, 'padding': 0} if init_phase else {'kernel_size': 4, 'stride': 2, 'padding': 1}
        self._nn.append(TransposableConvTranspose2d(hidden_nc1, hidden_nc2, **convt_kwargs, bias=True))
        self._nn.append(InvElementwiseReLU())
        # 3. Channel changer. (hidden_nc2, input_size, input_size) -> (input_nc//2, input_size, input_size)
        self._nn.append(InvConv2d1x1Fixed(hidden_nc2, input_nc//2))

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
    def __init__(self, input_nc, input_size, hidden_nc1, hidden_nc2, **kwargs):
        """
        :param input_nc:
        :param input_size:
        :param hidden_nc1:
        :param hidden_nc2:
        :param kwargs:
            - ds_conv: For init layer.
        :return:
        """
        super(Flow, self).__init__()
        # Config.
        self._init_phase = True if 'ds_conv' in kwargs.keys() else False
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Conv.
        if not self._init_phase:
            self._conv = InvConv2d1x1Fixed(input_nc, input_nc)
        # 2. Coupling.
        self._coupling = Coupling(input_nc, input_size, hidden_nc1, hidden_nc2, **kwargs)

    # ------------------------------------------------------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------------------------------------------------------

    def forward(self, x, linearize=False):
        # 1. Forward.
        if not self._init_phase:
            x = self._conv(x)
        output = self._coupling(x, linearize=linearize)
        # Return
        return output

    def linearized_transpose(self, x, retain_grads=False):
        output = self._coupling.linearized_transpose(x, retain_grads=retain_grads)
        if not self._init_phase:
            output = self._conv.linearized_transpose(output)
        # Return
        return output


class Block(BaseModule):
    """
    Block.
    """
    def __init__(self, input_nc, input_size, hidden_nc1, hidden_nc2, **kwargs):
        """
        :param input_nc:
        :param input_size:
        :param final_conv:
        :param kwargs
             - ds_conv: For init layer.
             - n_flows: For middle layers.
        """
        super(Block, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        # Flows.
        self._flows = nn.ModuleList([])
        for _ in range(1 if 'ds_conv' in kwargs.keys() else kwargs.pop('n_flows')):
            self._flows.append(Flow(input_nc, input_size, hidden_nc1, hidden_nc2, **kwargs))

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


class InitDiuBlock(BaseModule):
    """
    Init DIU + Block module.
    """
    def __init__(self, nz, output_nc, output_size, hidden_nc):
        super(InitDiuBlock, self).__init__()
        assert output_nc % 2 == 0 and output_nc > nz*2
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Conv & usqz & concat0.
        self._conv = InvConv2d1x1Fixed(nz, (output_nc//2)*(output_size**2))
        self._usqz = Unsqueeze(s=output_size)
        self._cat = ConcatZero()
        # 2. Block.
        self._block = Block(
            output_nc, output_size, hidden_nc1=nz, hidden_nc2=hidden_nc,
            ds_conv=InvConv2d1x1Fixed((output_nc//2)*(output_size**2), nz, matrix_r=self._conv._matrix_r.T))

    def forward(self, x, linearize=False):
        # 1. DIU.
        output = self._conv(x)
        output = self._usqz(output)
        output = self._cat(output)
        # 2. Block.
        ret = self._block(output, linearize=linearize)
        # Return
        return ret

    def linearized_transpose(self, x, retain_grads=False):
        # 1. Block.
        output = self._block.linearized_transpose(x, retain_grads=retain_grads)
        # 2. DIU.
        output = self._cat.linearized_transpose(output)
        output = self._usqz.linearized_transpose(output)
        output = self._conv.linearized_transpose(output)
        # Return
        return output


########################################################################################################################
# The flow-decoder.
########################################################################################################################

class FlowDecoder(BaseModule):
    """
    Decoder.
    """
    def __init__(self, nz, init_size, img_size, ncs, hidden_ncs, middle_ns_flows):
        """
        For example:
            - ncs:                  128(4,4),   64(8,8),   32(16,16),  16(32,32),   4(64,64)
            - hidden_ncs:           256,        128,       64,         32,          16
            - middle_ns_flows:                  3,         3,          3,           3
        """
        super(FlowDecoder, self).__init__()
        # Configs & check.
        n_layers = int(math.log2(img_size//init_size))
        assert init_size*2**n_layers == img_size
        ################################################################################################################
        # Architecture.
        ################################################################################################################
        # 1. Init layer. (nz, 1, 1) -> (ncs[0], init_size, init_size)
        self._init_diu_block = InitDiuBlock(nz, ncs[0], init_size, hidden_ncs[0])
        # 2. Middle layers. (ncs[i], init_size*2^i, init_size*2^i) -> (ncs[i+1], init_size*2^(i+1), init_size*2^(i+1))
        self._middle_dius, self._middle_blocks = nn.ModuleList([]), nn.ModuleList([])
        for index in range(n_layers):
            # (1) DimIncreaseUnsqueeze2.
            self._middle_dius.append(DimIncreaseUnsqueeze2(ncs[index], ncs[index+1]))
            # (2) Block.
            self._middle_blocks.append(Block(
                ncs[index+1], init_size*2**(index+1), hidden_ncs[index], hidden_ncs[index+1], n_flows=middle_ns_flows[index]))
        # 3. Final conv.
        self._final_conv = InvConv2d1x1Fixed(ncs[-1], ncs[-1])

    def forward(self, z, linearize=False):
        # 1. Init layer.
        output = self._init_diu_block(z.unsqueeze(-1).unsqueeze(-1), linearize=linearize)
        # 2. Middle layers.
        for middle_diu, middle_block in zip(self._middle_dius, self._middle_blocks):
            output = middle_diu(output)
            output = middle_block(output, linearize=linearize)
        # 3. Final.
        output = self._final_conv(output)
        # Return
        return output

    def linearized_transpose(self, x, retain_grads=False):
        # 1. Final.
        output = self._final_conv.linearized_transpose(x)
        # 2. Middle layers.
        for middle_diu, middle_block in zip(self._middle_dius[::-1], self._middle_blocks[::-1]):
            output = middle_block.linearized_transpose(output, retain_grads=retain_grads)
            output = middle_diu.linearized_transpose(output)
        # 3. Init layer.
        output = self._init_diu_block.linearized_transpose(output, retain_grads=retain_grads)
        # Return
        return output.squeeze(-1).squeeze(-1)
