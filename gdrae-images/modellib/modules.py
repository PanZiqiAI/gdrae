
from torch import nn
from torch.nn import init
from functools import partial
from utils.operations import *
from custom_pkg.pytorch.operations import reparameterize
from modellib.components import FlowDecoder, InvElementwiseSigmoid


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
    def __init__(self, nz, sigma):
        super(Encoder, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Architecture
        # --------------------------------------------------------------------------------------------------------------
        self._module = self._get_module(nz)
        self.register_buffer('_logvar', torch.tensor([sigma]).log()*2.0)
        # --------------------------------------------------------------------------------------------------------------
        # Init weight
        # --------------------------------------------------------------------------------------------------------------
        self.apply(init_weights)

    def _get_module(self, nz):
        raise NotImplementedError

    def forward(self, x):
        # Get params.
        mu = self._module(x).squeeze(-1).squeeze(-1)
        logvar = self._logvar.unsqueeze(0).expand(*mu.size())
        # Resampling
        z = reparameterize(mu, logvar)
        # Return
        return z, (mu, logvar)


class Encoder64x64x1Simple(Encoder):
    """
    Encoder 64x64x1.
    """
    def _get_module(self, nz):
        return nn.Sequential(
            # (1, 64, 64) -> (32, 32, 32) -> (32, 16, 16)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            # (32, 16, 16) -> (64, 8, 8) -> (64, 4, 4)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            # (64, 4, 4) -> (128, 1, 1) -> (2*nz, 1, 1)
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0), nn.ReLU(True),
            nn.Conv2d(128, nz, kernel_size=1))


class Encoder64x64x1Complex(Encoder):
    """
    Encoder 64x64x1.
    """
    def _get_module(self, nz):
        return nn.Sequential(
            # (1, 64, 64) -> (32, 32, 32) -> (32, 16, 16)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            # (32, 16, 16) -> (64, 8, 8) -> (64, 4, 4)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            # (64, 4, 4) -> (256, 1, 1) -> (2*nz, 1, 1)
            nn.Conv2d(64, 256, kernel_size=4, stride=1, padding=0), nn.ReLU(True),
            nn.Conv2d(256, nz, kernel_size=1))


class Encoder64x64x3(Encoder):
    """
    Encoder 64x64x3.
    """
    def _get_module(self, nz):
        return nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32) -> (32, 16, 16)
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            # (32, 16, 16) -> (64, 8, 8) -> (64, 4, 4)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            # (64, 4, 4) -> (256, 1, 1) -> (2*nz, 1, 1)
            nn.Conv2d(64, 256, kernel_size=4, stride=1, padding=0), nn.ReLU(True),
            nn.Conv2d(256, nz, kernel_size=1))


class SVPredictor(nn.Module):
    """
    Singular value predictor.
    """
    def __init__(self, nz):
        super(SVPredictor, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Architecture
        # --------------------------------------------------------------------------------------------------------------
        self._module = nn.Sequential(
            # (nz, ) -> (1024, )
            nn.Linear(nz, 1024), nn.ReLU(True),
            # (1024, ) -> (1024, )
            nn.Linear(1024, 1024), nn.ReLU(True),
            # (1024, ) -> (nz, )
            nn.Linear(1024, nz-1))
        # --------------------------------------------------------------------------------------------------------------
        # Init weight
        # --------------------------------------------------------------------------------------------------------------
        self.apply(init_weights)

    def forward(self, z):
        # Forward.
        logs = self._module(z)
        # 1. Get logs.
        logs = torch.cat([logs, -logs.sum(dim=1, keepdim=True)], dim=1)
        # Return
        return logs


########################################################################################################################
# Decoder
########################################################################################################################

class Decoder(nn.Module):
    """
    Decoder.
    """
    def __init__(self, nz, init_size, img_nc, img_size, ncs, hidden_ncs, middle_ns_flows):
        super(Decoder, self).__init__()
        # Config.
        self._img_nc = img_nc
        ################################################################################################################
        # Architecture.
        ################################################################################################################
        # 1. Decoder.
        self.register_parameter("_param_log_capacity", nn.Parameter(torch.zeros(1)))
        self._flow_dec = FlowDecoder(
            nz=nz, init_size=init_size, img_size=img_size, ncs=ncs, hidden_ncs=hidden_ncs, middle_ns_flows=middle_ns_flows)
        self._final_sigmoid = InvElementwiseSigmoid()
        # 2. Jacob singular values.
        self._logsv_pred = SVPredictor(nz)
        ################################################################################################################
        # Init weights.
        ################################################################################################################
        self.apply(init_weights)

    def forward(self, z, apply_sigmoid=True):
        """ For generating image. """
        output = self.log_capacity.exp() * z
        output = self._flow_dec(output)[:, :self._img_nc]
        if apply_sigmoid: output = self._final_sigmoid(output)
        # Return
        return output

    # ------------------------------------------------------------------------------------------------------------------
    # sv norms: capacity + sv.
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def log_capacity(self):
        return self._param_log_capacity

    def calc_sv(self, z):
        param_z = self.log_capacity.exp() * z
        param_s = self._logsv_pred(param_z).exp()
        # Return
        return param_s

    # ------------------------------------------------------------------------------------------------------------------
    # Jacob: orthogonal (sn) & distinct (sv)
    # ------------------------------------------------------------------------------------------------------------------

    def _call_fw(self, e, param_z, param_s_r, linearize=False):
        """ Forward method w.r.t. e. """
        output = (param_z - param_s_r).detach() + param_s_r*e
        output = self._flow_dec(output, linearize=linearize)
        output = self._final_sigmoid(output, linearize=linearize)
        # Return
        return output

    def _call_lt(self, eps, param_s_r):
        """ This LT is with respect to e. """
        output = self._final_sigmoid.linearized_transpose(eps)
        output = self._flow_dec.linearized_transpose(output)
        output = param_s_r * output
        # Return
        return output

    def calc_logsn(self, z, sn_power=3):
        param_z = self.log_capacity.exp() * z
        param_logs = self._logsv_pred(param_z)
        param_s_r = (-param_logs).exp()
        ################################################################################################################
        # Forward.
        ################################################################################################################
        # 1. Forward.
        e = torch.ones_like(param_z, requires_grad=True)
        output = self._call_fw(e, param_z, param_s_r, linearize=True)
        # 2. LT.
        x_t = torch.randn_like(output).requires_grad_(True)
        output_t = self._call_lt(x_t, param_s_r)
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
        return {'logsn': sn.log(), 'logsv': param_logs}

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluation.
    # ------------------------------------------------------------------------------------------------------------------

    @api_empty_cache
    def calc_lt_err(self, z, jacob_size=16):
        param_z = self.log_capacity.exp() * z
        param_s_r = (-self._logsv_pred(param_z)).exp()
        # Linearing
        with torch.no_grad(): output = self._call_fw(torch.ones_like(param_z), param_z, param_s_r, linearize=True)
        # 1. Calculate jacob & jacob_t.
        _j_param_z, _j_param_s_r = map(lambda _x: _x.unsqueeze(1).expand(_x.size(0), jacob_size, *_x.size()[1:]).reshape(-1, *_x.size()[1:]), [param_z, param_s_r])
        # (1) Get jacob.
        jacob = autograd_jacob(torch.ones_like(param_z), partial(self._call_fw, param_z=_j_param_z, param_s_r=_j_param_s_r), bsize=jacob_size)
        # (2) Get jacob_t.
        jacob_t = autograd_jacob(torch.randn_like(output), partial(self._call_lt, param_s_r=_j_param_s_r), bsize=jacob_size)
        # 2. Get results.
        err = normalized_mean_absolute_err(jacob_t, jacob.transpose(1, 2))
        # Return
        return err

    @api_empty_cache
    def calc_jacob_ortho(self, z, jacob_size=16):
        param_z = self.log_capacity.exp() * z
        param_s = self._logsv_pred(param_z).exp()
        # 1. Calculate jacob & jtj.
        # (1) Get jacob.
        _j_param_z = param_z.unsqueeze(1).expand(param_z.size(0), jacob_size, *param_z.size()[1:]).reshape(-1, *param_z.size()[1:])
        jacob = autograd_jacob(torch.ones_like(param_z), partial(self._call_fw, param_z=_j_param_z, param_s_r=torch.ones_like(_j_param_z)), bsize=jacob_size)
        # (2) Get jtj.
        jtj = torch.matmul(jacob.transpose(1, 2), jacob)
        # 2. Get results.
        ortho_sign, elem_diag = measure_orthogonality(jtj)
        err = normalized_mean_absolute_err(param_s, elem_diag.sqrt())
        # Return
        return {'ortho_sign': ortho_sign.mean().item(), 'sv_match_err': err}
