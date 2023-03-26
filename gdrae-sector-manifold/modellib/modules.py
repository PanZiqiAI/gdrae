
from torch import nn
from torch.nn import init
from functools import partial
from utils.operations import *
from _flow_decoder import InvConv2d1x1Fixed
from _flow_decoder import Decoder as FlowDecoder


def init_kaiming(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None: m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None: m.bias.data.fill_(0)


def init_normal(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None: m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None: m.bias.data.fill_(0)


def init_weights(m, mode='normal'):
    if mode == 'kaiming': initializer = init_kaiming
    elif mode == 'normal': initializer = init_normal
    else: raise ValueError
    # Init
    initializer(m)


########################################################################################################################
# Encoder
########################################################################################################################

class Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, input_nc, output_nc):
        super(Encoder, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Architecture
        # --------------------------------------------------------------------------------------------------------------
        self._module = nn.Sequential(
            # (input_nc, ) -> (1024, )
            nn.Linear(input_nc, 1024), nn.ReLU(True),
            # (1024, ) -> (1024, )
            nn.Linear(1024, 1024), nn.ReLU(True),
            # (1024, ) -> (output_nc, )
            nn.Linear(1024, output_nc))
        # --------------------------------------------------------------------------------------------------------------
        # Init weight
        # --------------------------------------------------------------------------------------------------------------
        self.apply(init_weights)

    def forward(self, x):
        return self._module(x)


########################################################################################################################
# Decoder
########################################################################################################################

class Decoder(nn.Module):
    """
    Decoder.
    """
    def __init__(self, nz, input_nc, hidden_nc, n_flows):
        super(Decoder, self).__init__()
        # Config.
        assert input_nc >= nz
        self._nz, self._input_nc = nz, input_nc
        ################################################################################################################
        # Architecture.
        ################################################################################################################
        # 1.1. Capacity.
        self.register_parameter("_param_log_capacity", nn.Parameter(torch.zeros(1)))
        # 1.2. Flow-decoder.
        self._conv = InvConv2d1x1Fixed(nz, input_nc)
        self._flow_dec = FlowDecoder(input_nc, hidden_nc, n_flows)
        # 2. Jacob singular values.
        self._enc_s = Encoder(input_nc=nz, output_nc=nz-1)
        ################################################################################################################
        # Init weights.
        ################################################################################################################
        self.apply(init_weights)

    @property
    def log_capacity(self):
        return self._param_log_capacity

    def forward(self, z):
        output = self.log_capacity.exp() * z
        output = self._conv(output.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        output = self._flow_dec(output)
        # Return
        if self._input_nc == self._nz:
            return {'output': output}
        else:
            output, res_feat = torch.split(output, split_size_or_sections=[self._nz, self._input_nc-self._nz], dim=1)
            return {'output': output, 'res_feat': res_feat}

    def _call_fw(self, e, param_z, param_s, linearize=False):
        """ Forward method w.r.t. e. """
        output = (param_z - param_s).detach() + param_s*e
        output = self._conv(output.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        output = self._flow_dec(output, linearize=linearize)
        # Return
        return output

    def _call_lt(self, eps, param_s):
        """ This LT is with respect to e. """
        output = self._flow_dec.linearized_transpose(eps)
        output = self._conv.linearized_transpose(output.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        output = param_s * output
        # Return
        return output

    def _get_logs(self, param_z):
        logs = self._enc_s(param_z)
        logs = torch.cat([logs, -logs.sum(dim=1, keepdim=True)], dim=1)
        # Return
        return logs

    def calc_logsn(self, z, sn_power=3):
        # Get param_z (applied capacity) & param_s.
        param_z = self.log_capacity.exp() * z
        param_logs = self._get_logs(param_z)
        param_s = param_logs.exp()
        ################################################################################################################
        # Forward.
        ################################################################################################################
        # 1. Forward.
        e = torch.ones_like(param_z, requires_grad=True)
        output = self._call_fw(e, param_z, param_s, linearize=True)
        # 2. LT.
        x_t = torch.randn_like(output).requires_grad_(True)
        output_t = self._call_lt(x_t, param_s)
        ################################################################################################################
        # Fast approximation.
        ################################################################################################################
        # (1) Init u & v.
        u, last_u, v = torch.randn_like(output), None, None
        # (2) Approximation iterations.
        for index in range(sn_power):
            # (1) Update v.
            v = autograd_proc(eps=l2_normalization(u), ipt=e, opt=output)
            # (2) Update u.
            u, last_u = autograd_proc(eps=l2_normalization(v), ipt=x_t, opt=output_t, create_graph=index == sn_power-1), u.detach()
        # (3) Get result.
        u, last_u = map(lambda _x: _x.reshape(_x.size(0), -1), [u, last_u])
        sn = (u*last_u).sum(dim=1)
        # Return
        return {'logsn': sn.log(), 'logsv': -param_logs}

    @api_empty_cache
    def calc_lt_err(self, z, jacob_size=16):
        # Get param_z (applied capacity) & param_s.
        param_z = self.log_capacity.exp() * z
        param_s = self._get_logs(param_z).exp()
        # Linearing
        with torch.no_grad(): output = self._call_fw(torch.ones_like(param_z), param_z, param_s, linearize=True)
        # 1. Calculate jacob & jacob_t.
        _j_param_z, _j_param_s = map(lambda _x: _x.unsqueeze(1).expand(_x.size(0), jacob_size, *_x.size()[1:]).reshape(-1, *_x.size()[1:]), [param_z, param_s])
        # (1) Get jacob.
        jacob = autograd_jacob(torch.ones_like(param_z), partial(self._call_fw, param_z=_j_param_z, param_s=_j_param_s), bsize=jacob_size)
        # (2) Get jacob_t.
        jacob_t = autograd_jacob(torch.randn_like(output), partial(self._call_lt, param_s=_j_param_s), bsize=jacob_size)
        # 2. Get results.
        err = normalized_mean_absolute_err(jacob_t, jacob.transpose(1, 2))
        # Return
        return err

    @api_empty_cache
    def calc_jacob_ortho(self, z, jacob_size=16):
        # Get param_z (applied capacity) & param_s.
        param_z = self.log_capacity.exp() * z
        param_s = self._get_logs(param_z).exp()
        # 1. Calculate jacob & jtj.
        # (1) Get jacob.
        _j_param_z = param_z.unsqueeze(1).expand(param_z.size(0), jacob_size, *param_z.size()[1:]).reshape(-1, *param_z.size()[1:])
        jacob = autograd_jacob(torch.ones_like(param_z), partial(self._call_fw, param_z=_j_param_z, param_s=torch.ones_like(_j_param_z)), bsize=jacob_size)
        # (2) Get jtj.
        jtj = torch.matmul(jacob.transpose(1, 2), jacob)
        # 2. Get results.
        ortho_sign, elem_diag = measure_orthogonality(jtj)
        err = normalized_mean_absolute_err(elem_diag.sqrt()*param_s, torch.ones_like(param_s))
        # Return
        return {'ortho_sign': ortho_sign.mean().item(), 'sv_match_err': err}
