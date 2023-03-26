
from tqdm import tqdm
from dlib import Metric
from modellib.modules import *
from utils.criterions import *
from utils.evaluations import *
from custom_pkg.io.logger import Logger
from torchvision.utils import save_image
from custom_pkg.pytorch.base_models import IterativeBaseModel
from custom_pkg.basic.operations import fet_d, PathPreparation, BatchSlicerInt
from custom_pkg.basic.metrics import FreqCounter, summarize_to_xls
from custom_pkg.pytorch.operations import summarize_losses_and_backward, KLDivLoss, sampling_z
from custom_pkg.basic.metrics import DisenMetricMIG, vis_latent_traversal, vis_latent_traversal_given_x


class Trainer(IterativeBaseModel):
    """ Trainer. """
    def _build_architectures(self):
        # Get encoder.
        if self._cfg.args.dataset == 'shapes2d':
            encoder = Encoder64x64x1Simple(nz=self._cfg.args.nz, sigma=self._cfg.args.sigma)
        elif self._cfg.args.dataset in ['faces', 'small_norb']:
            encoder = Encoder64x64x1Complex(nz=self._cfg.args.nz, sigma=self._cfg.args.sigma)
        elif self._cfg.args.dataset in ['shapes3d', 'cars3d']:
            encoder = Encoder64x64x3(nz=self._cfg.args.nz, sigma=self._cfg.args.sigma)
        else: raise ValueError
        # Get decoder.
        decoder = Decoder(
            nz=self._cfg.args.nz, init_size=self._cfg.args.init_size, img_nc=self._cfg.args.input_nc, img_size=self._cfg.args.input_size,
            ncs=self._cfg.args.ncs, hidden_ncs=self._cfg.args.hidden_ncs, middle_ns_flows=self._cfg.args.middle_ns_flows)
        """ Init """
        super(Trainer, self)._build_architectures(Enc=encoder, Dec=decoder)

    def _save_checkpoint(self, n, stale_n=None, **kwargs):
        super(Trainer, self)._save_checkpoint(n, stale_n, items={
            'meter-%s' % k: self._meters[k] for k in ['records_eval_quant_my', 'records_eval_quant_dlib']})

    def _resume(self, mode, checkpoint_pack=None, verbose=False, **kwargs):

        def _load_meter_items(_chkpt):
            for k in ['records_eval_quant_my', 'records_eval_quant_dlib']:
                self._meters[k] = _chkpt['meter-%s' % k]

        super(Trainer, self)._resume(mode, checkpoint_pack, verbose, lmd_load_meter_items=_load_meter_items)

    def _set_logs(self, **kwargs):
        super(Trainer, self)._set_logs(**kwargs)
        # --------------------------------------------------------------------------------------------------------------
        # Quantitative evaluation.
        # --------------------------------------------------------------------------------------------------------------
        # (1) My implementation.
        for k in ['mig', 'jemmig', 'migsup', 'modularity_score', 'dcimig']:
            self._logs['log_eval_quant_my_%s' % k] = Logger(
                self._cfg.args.eval_quant_dir, 'my_%s' % k, formatted_prefix=self._cfg.args.desc, formatted_counters=['epoch', 'batch', 'step', 'iter'],
                append_mode=False if self._cfg.args.load_from == -1 else True, pbar=self._meters['pbar'])
        # (2) Disentanglement lib implementation.
        self._logs['log_eval_quant_dlib'] = Logger(
            self._cfg.args.eval_quant_dir, 'dlib', formatted_prefix=self._cfg.args.desc, formatted_counters=['epoch', 'batch', 'step', 'iter'],
            append_mode=False if self._cfg.args.load_from == -1 else True, pbar=self._meters['pbar'])
        # (3) Jacob.
        self._logs['log_eval_jacob'] = Logger(
            self._cfg.args.ana_train_dir, 'eval_jacob', formatted_prefix=self._cfg.args.desc, formatted_counters=['epoch', 'batch', 'step', 'iter'],
            append_mode=False if self._cfg.args.load_from == -1 else True, pbar=self._meters['pbar'])

    def _set_criterions(self):
        # Reconstruction
        self._criterions['recon'] = ReconBCELoss()
        # KLD
        self._criterions['kld'] = KLDivLoss(random_type='uni', random_uni_radius=self._cfg.args.random_uni_radius)
        # Jacob & norm.
        self._criterions['jacob'] = LossJacob()
        self._criterions['norm'] = LossNorm()

    def _set_optimizers(self):
        self._optimizers['vae'] = torch.optim.Adam(self.parameters(), lr=self._cfg.args.learning_rate, betas=(0.9, 0.999))

    def _set_meters(self, **kwargs):
        super(Trainer, self)._set_meters()
        # Evaluation.
        self._meters['counter_eval_quali'] = FreqCounter(
            self._cfg.args.freq_step_eval_quali, iter_fetcher=lambda: self._meters['i']['step'])
        self._meters['counter_eval_quant'] = FreqCounter(
            self._cfg.args.freq_step_eval_quant, iter_fetcher=lambda: self._meters['i']['step'])
        self._meters['counter_eval_jacob'] = FreqCounter(
            self._cfg.args.freq_step_eval_jacob, iter_fetcher=lambda: self._meters['i']['step'])
        self._meters['records_eval_quant_my'] = []
        self._meters['records_eval_quant_dlib'] = []

    def _init_packs(self):
        return super(Trainer, self)._init_packs('quali', 'meters')

    ####################################################################################################################
    # Training
    ####################################################################################################################

    def _deploy_batch_data(self, batch_data):
        images = batch_data.to(self._cfg.args.device)
        return images.size(0), images

    def _sampling_z(self, batch_size):
        return sampling_z(batch_size, self._cfg.args.nz, device=self._cfg.args.device, random_type='uni', random_uni_radius=self._cfg.args.random_uni_radius)

    def _train_step(self, packs):
        # --------------------------------------------------------------------------------------------------------------
        # Forward.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Reconstruction.
        x_real = self._fetch_batch_data()
        z_enc, (mu, _) = self._Enc(x_real)
        x_recon = self._Dec(z_enc, apply_sigmoid=False)
        # 2. Jacob (orthogonal + distinct sv.) & norms (capacity + sv.)
        z_rnd = self._sampling_z(self._cfg.args.batch_size)
        # (1) Jacob.
        jacob = self._Dec.calc_logsn(z_rnd, sn_power=self._cfg.args.sn_power)
        # (2) Norm.
        norm_capacity = self._Dec.log_capacity
        norm_sv = self._Dec.calc_sv(z_rnd)
        # --------------------------------------------------------------------------------------------------------------
        # Calculate loss.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Reconstruction & KLD.
        losses = {
            'loss_recon': self._criterions['recon'](x_real, x_recon, lmd=self._cfg.args.lambda_recon),
            'loss_kld': self._criterions['kld'](mu, lmd=self._cfg.args.lambda_kld)}
        # 2. Jacob (orthogonal + distinct sv.) & norms (capacity + sv.)
        losses.update(self._criterions['jacob'](
            jacob, lmd={'loss_jacob_sn': self._cfg.args.lambda_jacob_sn, 'loss_jacob_sv': self._cfg.args.lambda_jacob_sv}))
        losses.update(self._criterions['norm'](
            norm_capacity, norm_sv, lmd={'loss_norm_capacity': self._cfg.args.lambda_norm_capacity, 'loss_norm_sv': self._cfg.args.lambda_norm_sv}))
        """ Backward & update. """
        self._optimizers['vae'].zero_grad()
        summarize_losses_and_backward(*losses.values())
        self._optimizers['vae'].step()
        # --------------------------------------------------------------------------------------------------------------
        """ Saving """
        # (1) Qualititative results.
        for k, v in zip(['x_real', 'x_recon'], [x_real.cpu(), x_recon.sigmoid().detach().cpu()]):
            packs['quali'][k] = (torch.cat([packs['quali'][k], v]) if k in packs['quali'] else v)[-self._cfg.args.eval_quali_recon_sqrt_num**2:]
        # (2) Logs.
        packs['meters'].update({'recon_l1err': self._criterions['recon'].recon_l1err(x_real, x_recon.sigmoid())})
        packs['meters'].update({'mu': mu.abs().mean(0).cpu()})
        packs['meters'].update({'sv': jacob['logsv'].exp().mean(0).cpu()})
        packs['log'].update({k: v.item() for k, v in losses.items()})

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
                'train/losses': fet_d(packs['log'], prefix='loss_', replace='')})

        super(Trainer, self)._process_log(packs, lmd_generate_log=_lmd_generate_log)

    def _process_after_step(self, packs, **kwargs):
        # 1. Logging.
        self._process_log(packs, **kwargs)
        # 2. Evaluation.
        if self._meters['counter_eval_quali'].check():
            self._eval_quali(packs)
        if self._meters['counter_eval_quant'].check():
            self._eval_quant_my()
            self._eval_quant_dlib()
        if self._meters['counter_eval_jacob'].check():
            self._eval_jacob()
        # 3. lr & chkpt.
        self._process_chkpt_and_lr()

    ####################################################################################################################
    # Evaluation.
    ####################################################################################################################

    @torch.no_grad()
    def _eval_quali(self, packs):
        filename, pad_value = 'step[%d]' % self._meters['i']['step'], (1 if self._cfg.args.dataset == 'shapes2d' else 0)
        x_real, x_recon = packs['quali']['x_real'], packs['quali']['x_recon']
        func_encode = lambda _x: self._Enc(_x)[1][0]
        func_decode = lambda _z: self._Dec(_z)
        # --------------------------------------------------------------------------------------------------------------
        # Reconstruction.
        # --------------------------------------------------------------------------------------------------------------
        x = torch.cat([x_real.unsqueeze(1), x_recon.unsqueeze(1)], dim=1)
        x = x.reshape(x.size(0)*2, *x.size()[2:])
        """ Saving """
        with PathPreparation(self._cfg.args.eval_quali_dir, 'recon', filename, ext='png') as (_, save_path):
            save_image(x, save_path, nrow=self._cfg.args.eval_quali_recon_sqrt_num*2, pad_value=pad_value)
        # --------------------------------------------------------------------------------------------------------------
        # Traversal: random real.
        # --------------------------------------------------------------------------------------------------------------
        results = vis_latent_traversal_given_x(
            func_encoder=func_encode, func_decoder=func_decode, x_real=x_real[:1].to(self._cfg.args.device),
            limit=self._cfg.args.random_uni_radius, n_traversals=self._cfg.args.eval_quali_trav_n)
        """ Saving """
        with PathPreparation(self._cfg.args.eval_quali_dir, 'trav-random_real', filename, ext='png') as (_, save_path):
            save_image(results.reshape(-1, *results.size()[-3:]), save_path, nrow=self._cfg.args.eval_quali_trav_n+2, pad_value=pad_value)
        # --------------------------------------------------------------------------------------------------------------
        # Traversal: random noise.
        # --------------------------------------------------------------------------------------------------------------
        results = vis_latent_traversal(
            func_decode, z=self._sampling_z(1), limit=self._cfg.args.random_uni_radius, n_traversals=self._cfg.args.eval_quali_trav_n)
        """ Saving """
        with PathPreparation(self._cfg.args.eval_quali_dir, 'trav-random_noise', filename, ext='png') as (_, save_path):
            save_image(results.reshape(-1, *results.size()[-3:]), save_path, nrow=self._cfg.args.eval_quali_trav_n+1, pad_value=pad_value)
        # --------------------------------------------------------------------------------------------------------------
        # Visualize singualr values.
        # --------------------------------------------------------------------------------------------------------------
        sv = []
        for batch_size in BatchSlicerInt(self._cfg.args.eval_quali_sv_n, self._cfg.args.eval_quali_sv_batch_size):
            sv.append(self._Dec.calc_sv(self._sampling_z(batch_size)).cpu())
        sv = torch.cat(sv).mean(0).numpy()
        """ Saving """
        with PathPreparation(self._cfg.args.eval_quali_dir, 'singular_values', filename, ext='png') as (_, save_path):
            vis_singular_values(sv, save_path)

    @torch.no_grad()
    def _eval_quant_my(self):
        # --------------------------------------------------------------------------------------------------------------
        # Get results.
        # --------------------------------------------------------------------------------------------------------------
        func_encode = lambda _x: self._Enc(_x)[1][0]
        eval_kwargs = {
            'ent_z_batch_size': self._cfg.args.eval_quant_my_z_batch_size, 'ent_z_max_counts': self._cfg.args.eval_quant_my_z_max_counts,
            'nz': self._cfg.args.nz, 'device': self._cfg.args.device}
        # 1. Init results. { metric1: { nbins1: score } }
        results = {}
        # 2. Evaluate.
        for nbins in self._cfg.args.eval_quant_my_z_nbins:
            evaluator = DisenMetricMIG(func_encoder=func_encode, ent_z_nbins=nbins, **eval_kwargs)
            cur_ret = evaluator(self._data['train'].dataset)
            """ Saving """
            for metric, scores in cur_ret.items():
                if metric not in results: results[metric] = {}
                results[metric][nbins] = scores
        # --------------------------------------------------------------------------------------------------------------
        # Logging.
        # --------------------------------------------------------------------------------------------------------------
        scores_record = {'step': self._meters['i']['step']}
        # 1. Log.
        for metric, scores in results.items():
            # Get items.
            items = {}
            for nbins, ret in scores.items():
                if isinstance(ret, dict): items.update({'nbins=%d@%s' % (nbins, k): v for k, v in ret.items()})
                else: items.update({'nbins=%d' % nbins: ret})
            # Log.
            self._logs['log_eval_quant_my_%s' % metric].info_formatted(counters=self._meters['i'], items=items, no_display_keys='all')
        # 2. TfBoard & setup record.
        for metric, scores in results.items():
            for nbins, ret in scores.items():
                # Tfboard.
                self._logs['tfboard'].add_multi_scalars(multi_scalars={
                    'eval_quant_my/%s' % metric: {'nbins=%d' % nbins: ret['_overall'] if isinstance(ret, dict) else ret}
                }, global_step=self._meters['i']['step'])
                # Record.
                scores_record['%s/nbins=%d' % (metric, nbins)] = ret['_overall'] if isinstance(ret, dict) else ret
        # Save record.
        self._meters['records_eval_quant_my'].append(scores_record)
        summarize_to_xls(self._meters['records_eval_quant_my'], self._cfg.args.eval_quant_dir, 'summary_my.xls')

    @torch.no_grad()
    def _eval_quant_dlib(self):
        func_encode = lambda _x: self._Enc(_x)[1][0]
        # Get results. { metric1: score, ... }
        evaluator = Metric(func_encode, device=self._cfg.args.device)
        results = evaluator(self._data['train'].dataset)
        # Reform results.
        scores = {}
        for metric, value in results.items():
            if isinstance(value, dict): scores.update({'%s/%s' % (metric, k): v for k, v in value.items()})
            else: scores[metric] = value
        # --------------------------------------------------------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------------------------------------------------------
        # 1. Log.
        self._logs['log_eval_quant_dlib'].info_formatted(counters=self._meters['i'], items=scores, no_display_keys='all')
        # 2. Tfboard
        self._logs['tfboard'].add_multi_scalars(multi_scalars={
            'eval_quant_dlib/%s' % metric: (v if isinstance(v, dict) else {'': v}) for metric, v in results.items()
        }, global_step=self._meters['i']['step'])
        # 3. Record.
        scores_record = {'step': self._meters['i']['step']}
        scores_record.update(scores)
        self._meters['records_eval_quant_dlib'].append(scores_record)
        summarize_to_xls(self._meters['records_eval_quant_dlib'], self._cfg.args.eval_quant_dir, 'summary_dlib.xls')

    def _eval_jacob(self):
        # 1. Init results.
        lt_err, jacob_ortho = [], {'ortho_sign': [], 'sv_match_err': []}
        # 2. Get results.
        pbar = tqdm(total=self._cfg.args.eval_jacob_n, desc="Evaluating Jacob")
        for batch_size in BatchSlicerInt(self._cfg.args.eval_jacob_n, self._cfg.args.eval_jacob_batch_size):
            z = self._sampling_z(batch_size)
            """ Get results. """
            lt_err.append(self._Dec.calc_lt_err(z, self._cfg.args.eval_jacob_ag_bsize))
            for k, v in self._Dec.calc_jacob_ortho(z, self._cfg.args.eval_jacob_ag_bsize).items(): jacob_ortho[k].append(v)
            """ Progress """
            pbar.update(batch_size)
        pbar.close()
        # 3. Concat.
        lt_err = np.array(lt_err).mean()
        jacob_ortho = {k: np.array(v).mean() for k, v in jacob_ortho.items()}
        """ Logging """
        self._logs['log_eval_jacob'].info_formatted(counters=self._meters['i'], items={
            'lt_err': lt_err, 'jacob_ortho_sign': jacob_ortho['ortho_sign'], 'jacob_sv_match_err': jacob_ortho['sv_match_err']})
        self._logs['tfboard'].add_multi_scalars(multi_scalars={
            'eval/jacob/lt_err': {'': lt_err},
            'eval/jacob/ortho': jacob_ortho
        }, global_step=self._meters['i']['step'])
