
import math
import torch
import numpy as np
from torch import nn
from torch.nn import init
from torch.autograd import grad
from ._custom_pkg.basic.metrics import FreqCounter
from ._custom_pkg.basic.visualizers import plt, gradient_colors
from ._custom_pkg.pytorch.base_models import IterativeBaseModel
from ._custom_pkg.basic.operations import fet_d, PathPreparation
from ._custom_pkg.pytorch.operations import BaseCriterion, summarize_losses_and_backward


########################################################################################################################
# Modules
########################################################################################################################

def init_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None: m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None: m.bias.data.fill_(0)


class Nonlinear(nn.Module):
    """
    Nonlinear function.
    """
    def __init__(self, input_nc, output_nc):
        super(Nonlinear, self).__init__()
        # --------------------------------------------------------------------------------------------------------------
        # Architecture.
        # --------------------------------------------------------------------------------------------------------------
        self._module = nn.Sequential(
            # (input_nc, ) -> (1024, )
            nn.Linear(input_nc, 1024), nn.ReLU(True),
            # (1024, ) -> (1024, )
            nn.Linear(1024, 1024), nn.ReLU(True),
            # (1024, ) -> (output_nc, )
            nn.Linear(1024, output_nc))
        # --------------------------------------------------------------------------------------------------------------
        # Init weight.
        # --------------------------------------------------------------------------------------------------------------
        self.apply(init_weight)

    def forward(self, x):
        return self._module(x)


class LogDet(nn.Module):
    """
    Logdet param.
    """
    def __init__(self):
        super(LogDet, self).__init__()
        # Param.
        self.register_parameter("_param_logdet", nn.Parameter(torch.zeros(1)))

    @property
    def logdet(self):
        return self._param_logdet


########################################################################################################################
# Utils.
########################################################################################################################

class ReconL2Loss(BaseCriterion):
    """ Reconstruction L2 Loss. """
    def _call_method(self, pred, gt):
        return ((pred - gt)**2).sum(1).mean()


class OnSectorLoss(BaseCriterion):
    """ On sector2d manifold loss. """
    def _call_method(self, x):
        # In region [0, 2] x [0, 2].
        loss_in_region = torch.relu((x - 1.0).abs() - 1.0).sum(1).mean()
        # In radius [1, 2].
        radius = (x**2).sum(dim=1, keepdim=True).sqrt()
        loss_in_radius = torch.relu((radius - 1.5).abs() - 0.5).sum(1).mean()
        # Return
        return loss_in_region + loss_in_radius


class OnLatentLoss(BaseCriterion):
    """ On latent loss. """
    def _call_method(self, z):
        # In radius [0, 1].
        loss = torch.relu((z - 0.5).abs() - 0.5).sum(1).mean()
        # Return
        return loss


class UnifLoss(BaseCriterion):
    """ Uniform loss. """
    def _call_method(self, z, x, logdet):
        # Calculate Jacobian.
        x0, x1 = x.chunk(2, dim=1)
        jacob = torch.cat([
            grad(outputs=_x, inputs=z, grad_outputs=torch.ones_like(_x), create_graph=True, retain_graph=True)[0].unsqueeze(1) for _x in [x0, x1]], dim=1)
        """ The Jacob should be orthogonal. Calculate logdet. """
        jtj = torch.matmul(jacob.transpose(1, 2), jacob)
        logdet_jacob = jtj[:, 0, 0].log() + jtj[:, 1, 1].log()
        # Calculate loss.
        loss = ((logdet_jacob - logdet)**2).mean()
        # Return
        return loss


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


def vis_rec_manifold(x_rec, save_path):
    """
    :param x_rec: (n, 2)
    :param save_path:
    :return:
    """
    # 1. Init figure.
    plt.figure(dpi=200)
    # 2. Scatters.
    plt.title("recovered manifold")
    plt.scatter(x=x_rec[:, 0], y=x_rec[:, 1], s=0.5)
    # Save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


########################################################################################################################
# Trainer
########################################################################################################################

class Trainer(IterativeBaseModel):
    """
    Trainer for function.
    """
    def _build_architectures(self):
        # Get f & encoder.
        f, encoder = Nonlinear(input_nc=1, output_nc=1), Nonlinear(input_nc=2, output_nc=2)
        """ Init """
        super(Trainer, self)._build_architectures(F=f, Enc=encoder, LogDet=LogDet())

    @property
    def f(self):
        return self._F.requires_grad_(False)

    def _set_criterions(self):
        self._criterions['recon'] = ReconL2Loss()
        self._criterions['om_sector'] = OnSectorLoss()
        self._criterions['om_latent'] = OnLatentLoss()
        self._criterions['unif'] = UnifLoss()

    def _set_optimizers(self):
        self._optimizers['main'] = torch.optim.Adam(self.parameters(), lr=self._cfg.args.learning_rate, betas=(0.9, 0.999))

    def _set_meters(self):
        super(Trainer, self)._set_meters()
        # Evaluation.
        self._meters['counter_eval_quali'] = FreqCounter(self._cfg.args.freq_step_eval_quali, iter_fetcher=lambda: self._meters['i']['step'])

    def _sampling_latent(self, batch_size):
        return torch.rand(size=(batch_size, 2), device=self._cfg.args.device)

    def _sampling_sector2d(self, batch_size):
        z = self._sampling_latent(batch_size)
        radius, angles = z.chunk(2, dim=1)
        radius, angles = radius+1.0, angles*(np.pi/2.0)
        data_x, data_y = radius*torch.cos(angles), radius*torch.sin(angles)
        # Return
        return torch.cat([data_x, data_y], dim=1)

    def _convert(self, z):
        radius, angles = z.chunk(2, dim=1)
        radius, angles = self._F(radius)+1.0, angles*(np.pi/2.0)
        data_x, data_y = radius*torch.cos(angles), radius*torch.sin(angles)
        # Return
        return torch.cat([data_x, data_y], dim=1)

    def _train_step(self, packs):
        losses = {}
        # --------------------------------------------------------------------------------------------------------------
        # 1. Latent -> Sector2d -> Latent.
        # --------------------------------------------------------------------------------------------------------------
        z = self._sampling_latent(self._cfg.args.batch_size).requires_grad_(True)
        x = self._convert(z)
        z_rec = self._Enc(x)
        """ Calculate losses. """
        losses['loss_2sector_recon'] = self._criterions['recon'](z_rec, z, lmd=self._cfg.args.lambda_2sector_recon)
        losses['loss_2sector_om'] = self._criterions['om_sector'](x, lmd=self._cfg.args.lambda_2sector_om)
        losses['loss_2sector_unif'] = self._criterions['unif'](z, x, self._LogDet.logdet, lmd=self._cfg.args.lambda_2sector_unif)
        # --------------------------------------------------------------------------------------------------------------
        # 2. Sector2d -> Latent -> Sector2d.
        # --------------------------------------------------------------------------------------------------------------
        x = self._sampling_sector2d(self._cfg.args.batch_size)
        z = self._Enc(x).requires_grad_(True)
        x_rec = self._convert(z)
        """ Calculate losses. """
        losses['loss_2latent_recon'] = self._criterions['recon'](x_rec, x, lmd=self._cfg.args.lambda_2latent_recon)
        losses['loss_2latent_om'] = self._criterions['om_latent'](z, lmd=self._cfg.args.lambda_2latent_om)
        losses['loss_2latent_unif'] = self._criterions['unif'](z, x_rec, self._LogDet.logdet, lmd=self._cfg.args.lambda_2latent_unif)
        """ Saving """
        packs['log'].update({k: v.item() for k, v in losses.items()})
        # --------------------------------------------------------------------------------------------------------------
        # Summarize & backward.
        self._optimizers['main'].zero_grad()
        summarize_losses_and_backward(*losses.values())
        self._optimizers['main'].step()

    def _process_log(self, packs, **kwargs):

        def _lmd_generate_log():
            packs['tfboard'].update({'train/losses': fet_d(packs['log'], prefix='loss_', replace='')})

        super(Trainer, self)._process_log(packs, lmd_generate_log=_lmd_generate_log)

    def _process_after_step(self, packs, **kwargs):
        super(Trainer, self)._process_after_step(packs, **kwargs)
        # Evaluation.
        if self._meters['counter_eval_quali'].check():
            self._eval_quali()

    @torch.no_grad()
    def _eval_quali(self):
        filename = 'step[%d]' % self._meters['i']['step']
        # --------------------------------------------------------------------------------------------------------------
        # Visualize correspondence.
        # --------------------------------------------------------------------------------------------------------------
        # (1) Get z.
        arange = torch.arange(0, 1.0+1e-8, step=1.0/(self._cfg.args.eval_quali_crsp_n_grids-1), device=self._cfg.args.device)
        z_dim1 = arange.unsqueeze(1).expand(len(arange), len(arange)).unsqueeze(-1)
        z_dim2 = arange.unsqueeze(0).expand(len(arange), len(arange)).unsqueeze(-1)
        z = torch.cat([z_dim1, z_dim2], dim=-1).reshape(-1, 2)
        # (2) Get x.
        x = self._convert(z)
        """ Visualize. """
        with PathPreparation(self._cfg.args.eval_quali_dir, 'correspondence', filename, ext='png') as (_, save_path):
            vis_correspondence(z.cpu().numpy(), x.cpu().numpy(), save_path)
        # --------------------------------------------------------------------------------------------------------------
        # Visualize random samples.
        # --------------------------------------------------------------------------------------------------------------
        # (1) Get z & x_rec.
        z = self._sampling_latent(batch_size=self._cfg.args.eval_quali_rec_n)
        x_rec = self._convert(z)
        """ Visualize. """
        with PathPreparation(self._cfg.args.eval_quali_dir, 'rec_manifold', filename, ext='png') as (_, save_path):
            vis_rec_manifold(x_rec.cpu().numpy(), save_path)
