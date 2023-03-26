
import numpy as np
from tqdm import tqdm
from modellib.modules import *
from utils.criterions import *
from utils.evaluations import *
from custom_pkg.io.logger import Logger
from custom_pkg.basic.metrics import FreqCounter
from custom_pkg.pytorch.base_models import IterativeBaseModel
from custom_pkg.basic.operations import fet_d, PathPreparation, BatchSlicerInt
from custom_pkg.pytorch.operations import summarize_losses_and_backward, sampling_z, KLDivLoss


class Trainer(IterativeBaseModel):
    """
    Trainer.
    """
    def _build_architectures(self):
        # Get encoder.
        encoder = Encoder(input_nc=2, output_nc=2)
        # Get decoder.
        decoder = Decoder(nz=2, input_nc=self._cfg.args.input_nc, hidden_nc=self._cfg.args.hidden_nc, n_flows=self._cfg.args.n_flows)
        """ Init """
        super(Trainer, self)._build_architectures(Enc=encoder, Dec=decoder)

    def _set_logs(self, **kwargs):
        super(Trainer, self)._set_logs(**kwargs)
        # Eval logger.
        self._logs['log_eval_quant'] = Logger(
            self._cfg.args.ana_train_dir, 'eval_quant', formatted_prefix=self._cfg.args.desc, formatted_counters=['epoch', 'batch', 'step', 'iter'],
            append_mode=False if self._cfg.args.load_from == -1 else True, pbar=self._meters['pbar'])

    def _set_criterions(self):
        # Reconstruction
        self._criterions['recon'] = ReconL2Loss()
        # KLD
        self._criterions['kld'] = KLDivLoss(random_type='uni', random_uni_radius=self._cfg.args.random_uni_radius)
        # Jacob.
        self._criterions['jacob'] = LossJacob()
        # Manifold compactness
        self._criterions['capacity'] = LossCapacity()

    def _set_optimizers(self):
        self._optimizers['vae'] = torch.optim.Adam(self.parameters(), lr=self._cfg.args.learning_rate, betas=(0.9, 0.999))

    def _set_meters(self):
        super(Trainer, self)._set_meters()
        # --------------------------------------------------------------------------------------------------------------
        # Evaluation.
        # --------------------------------------------------------------------------------------------------------------
        # Qualitative.
        self._meters['counter_eval_quali'] = FreqCounter(self._cfg.args.freq_step_eval_quali, iter_fetcher=lambda: self._meters['i']['step'])
        # Quantitative.
        self._meters['counter_eval_quant'] = FreqCounter(self._cfg.args.freq_step_eval_quant, iter_fetcher=lambda: self._meters['i']['step'])

    ####################################################################################################################
    # Train
    ####################################################################################################################

    def _deploy_batch_data(self, batch_data):
        images = batch_data.to(self._cfg.args.device)
        return images.size(0), images

    def _init_packs(self):
        return super(Trainer, self)._init_packs('meters')

    def _train_step(self, packs):
        # --------------------------------------------------------------------------------------------------------------
        # Forward.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Reconstruction.
        x_real = self._data['train'].dataset.sampling(self._cfg.args.batch_size)
        mu = self._Enc(x_real)
        x_recon = self._Dec(mu)
        # 2. Get Jacob.
        jacob = self._Dec.calc_logsn(self._sampling_z(self._cfg.args.batch_size), sn_power=self._cfg.args.sn_power)
        # --------------------------------------------------------------------------------------------------------------
        # Calculate loss.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Reconstruction & KLD.
        losses = {
            'loss_recon': self._criterions['recon'](x_real, x_recon['output'], lmd=self._cfg.args.lambda_recon),
            'loss_kld': self._criterions['kld'](mu, lmd=self._cfg.args.lambda_kld)
        }
        # 2. Jacob.
        losses.update(self._criterions['jacob'](jacob, lmd={'loss_jacob_sn': self._cfg.args.lambda_jacob_sn, 'loss_jacob_sv': self._cfg.args.lambda_jacob_sv}))
        # 3. Capacity.
        losses['loss_capacity'] = self._criterions['capacity'](self._Dec.log_capacity, lmd=self._cfg.args.lambda_capacity)
        """ Saving """
        packs['meters'].update({'recon_l1err': self._criterions['recon'].recon_l1err(x_real, x_recon['output'])})
        packs['meters'].update({k: v.abs().mean(0).cpu() for k, v in zip(['mu', 'sv'], [mu, jacob['logsv'].exp()])})
        packs['log'].update({k: v.item() for k, v in losses.items()})
        # --------------------------------------------------------------------------------------------------------------
        # Summarize & backward.
        self._optimizers['vae'].zero_grad()
        summarize_losses_and_backward(*losses.values())
        self._optimizers['vae'].step()

    def _process_log(self, packs, **kwargs):

        def _lmd_generate_log():
            # 1. Log.
            packs['log'].update(fet_d(packs['meters'], prefix='recon_'))
            packs['log'].update(fet_d(packs['meters'], 'mu', 'sv', lmd_v=lambda _v: _v.mean().item()))
            # 2. Tfboard.
            packs['tfboard'].update({
                'train/recon': fet_d(packs['meters'], prefix='recon_', replace=''),
                'train/mu': {'dim%d' % i: v.item() for i, v in enumerate(packs['meters']['mu'])},
                'train/sv': {'dim%d' % i: v.item() for i, v in enumerate(packs['meters']['sv'])},
                'train/losses': fet_d(packs['log'], prefix='loss_', replace='')
            })

        super(Trainer, self)._process_log(packs, lmd_generate_log=_lmd_generate_log)

    def _process_after_step(self, packs, **kwargs):
        # 1. Logging & chkpt & lr.
        super(Trainer, self)._process_after_step(packs, **kwargs)
        # 2. Evaluation
        if self._meters['counter_eval_quali'].check():
            self._eval_quali()
        if self._meters['counter_eval_quant'].check():
            self._eval_quant()

    ####################################################################################################################
    # Evaluation.
    ####################################################################################################################

    def _sampling_z(self, batch_size):
        return sampling_z(batch_size, 2, device=self._cfg.args.device, random_type='uni', random_uni_radius=self._cfg.args.random_uni_radius)

    @torch.no_grad()
    def _eval_quali(self):
        filename = 'step[%d]' % self._meters['i']['step']
        # --------------------------------------------------------------------------------------------------------------
        # Visualize correspondence
        # --------------------------------------------------------------------------------------------------------------
        # (1) Get z.
        limit = self._cfg.args.random_uni_radius
        arange = torch.arange(-limit, limit+1e-8, step=2.0*limit/(self._cfg.args.eval_quali_crsp_n_grids-1), device=self._cfg.args.device)
        z_dim1 = arange.unsqueeze(1).expand(len(arange), len(arange)).unsqueeze(-1)
        z_dim2 = arange.unsqueeze(0).expand(len(arange), len(arange)).unsqueeze(-1)
        z = torch.cat([z_dim1, z_dim2], dim=-1).reshape(-1, 2)
        # (2) Get x.
        x = self._Dec(z)['output']
        """ Visualize. """
        with PathPreparation(self._cfg.args.eval_quali_dir, 'correspondence', filename, ext='png') as (_, save_path):
            vis_correspondence(z.cpu().numpy(), x.cpu().numpy(), save_path)
        # --------------------------------------------------------------------------------------------------------------
        # Visualize mu scatter
        # --------------------------------------------------------------------------------------------------------------
        mu = []
        for batch_size in BatchSlicerInt(self._cfg.args.eval_quali_mu_n, 1024):
            batch_x = self._data['train'].dataset.sampling(batch_size)
            mu.append(self._Enc(batch_x).cpu())
        mu = torch.cat(mu).numpy()
        """ Visualize """
        with PathPreparation(self._cfg.args.eval_quali_dir, 'mu_scatter', filename, ext='png') as (_, save_path):
            vis_mu_scatter(mu, save_path)

    def _eval_quant(self):
        # 1. Init results.
        lt_err, jacob_ortho = [], {'ortho_sign': [], 'sv_match_err': []}
        # 2. Get results.
        pbar = tqdm(total=self._cfg.args.eval_quant_n, desc="Evaluating quantitative lt_err & jacob_ortho")
        for batch_size in BatchSlicerInt(self._cfg.args.eval_quant_n, 1024):
            z = self._sampling_z(batch_size)
            """ Get results. """
            lt_err.append(self._Dec.calc_lt_err(z, self._cfg.args.eval_quant_jacob_size))
            for k, v in self._Dec.calc_jacob_ortho(z, self._cfg.args.eval_quant_jacob_size).items(): jacob_ortho[k].append(v)
            """ Progress """
            pbar.update(batch_size)
        pbar.close()
        # 3. Concat.
        lt_err = np.array(lt_err).mean()
        jacob_ortho = {k: np.array(v).mean() for k, v in jacob_ortho.items()}
        """ Logging """
        self._logs['log_eval_quant'].info_formatted(counters=self._meters['i'], items={
            'lt_err': lt_err, 'jacob_ortho_sign': jacob_ortho['ortho_sign'], 'jacob_sv_match_err': jacob_ortho['sv_match_err']})
        self._logs['tfboard'].add_multi_scalars(multi_scalars={
            'eval/lt_err': {'decoder': lt_err},
            'eval/jacob_ortho': jacob_ortho
        }, global_step=self._meters['i']['step'])
