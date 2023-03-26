
import torch
from torch.nn import functional as F
from custom_pkg.pytorch.operations import BaseCriterion, nanmean


########################################################################################################################
# Reconstruction.
########################################################################################################################

class ReconBCELoss(BaseCriterion):
    """
    Reconstruction loss with binary cross entropy.
    """
    @staticmethod
    def recon_l1err(x_real, x_recon):
        return (x_recon - x_real).abs().mean().item()

    def _call_method(self, x_real, x_recon):
        loss = F.binary_cross_entropy_with_logits(x_recon, x_real, reduction='sum').div(x_real.size(0))
        return loss


########################################################################################################################
# Jacob orthogonal & singular value distinction
########################################################################################################################

class LossJacob(BaseCriterion):
    """
    Loss for jacob orthogonal & singular value distinction.
    """
    def _call_method(self, jacob_dict):
        logsn, logsv = jacob_dict['logsn'], jacob_dict['logsv']
        # 1. Jacob orthogonal.
        loss_sn = nanmean(logsn**2)
        # 2. Singular value distinction.
        sv_normed = torch.softmax(logsv, dim=1)
        loss_sv = (- sv_normed * torch.log(sv_normed + 1e-8)).sum(dim=1).mean()
        # Return
        return {'loss_jacob_sn': loss_sn, 'loss_jacob_sv': loss_sv}


########################################################################################################################
# Manifold compactness (singular value norms)
########################################################################################################################

class LossNorm(BaseCriterion):
    """
    Norm loss.
    """
    def _call_method(self, log_capacity, sv):
        """
        :param log_capacity: (1, )
        :param sv: (nz, )
        :return:
        """
        # 1. Capacity
        loss_capacity = log_capacity.sum()
        # 2. Predicted sv.
        loss_sv = (sv**2).mean()
        # Return
        return {'loss_norm_capacity': loss_capacity, 'loss_norm_sv': loss_sv}
