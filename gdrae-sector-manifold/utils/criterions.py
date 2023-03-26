
import torch
from custom_pkg.pytorch.operations import BaseCriterion, nanmean


########################################################################################################################
# Reconstruction.
########################################################################################################################

class ReconL2Loss(BaseCriterion):
    """
    Reconstruction loss with binary cross entropy.
    """
    @staticmethod
    def recon_l1err(x_real, x_recon):
        return (x_recon - x_real).abs().mean().item()

    def _call_method(self, x_real, x_recon):
        # Get loss.
        loss = ((x_real - x_recon)**2).sum(1).mean()
        # Return
        return loss


########################################################################################################################
# Manifold compactness
########################################################################################################################

class LossCapacity(BaseCriterion):
    """
    Capacity loss.
    """
    def _call_method(self, capacity):
        return capacity.sum()


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
