
import os
import math
import time
import xlwt
import torch
import atexit
import numpy as np
from tqdm import tqdm
from functools import wraps
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from ..basic.operations import chk_d, TempKwargsManager, BatchSlicerLenObj
from ..pytorch.operations import reparameterize, gaussian_log_density_marginal, gaussian_cross_entropy_marginal


########################################################################################################################
# Meters
########################################################################################################################

class ResumableMeter(object):
    """
    Meter that is resumable.
    """
    def load(self, **kwargs):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


class BestPerfMeter(ResumableMeter):
    """
    Meter to remember the best.
    """
    def __init__(self, iter_name, perf_name, lmd_ascend_perf=lambda new, stale: new > stale, early_stop_trials=-1):
        # Configuration
        self._iter_name, self._perf_name = iter_name, perf_name
        self._lmd_ascend_perf = lmd_ascend_perf
        self._early_stop_trials = early_stop_trials
        # Data
        self._best_iter = None
        self._best_perf = None
        self._trials_no_ascend = 0

    def load(self, **kwargs):
        self._best_iter = kwargs['best_%s' % self._iter_name]
        self._best_perf = kwargs['best_%s' % self._perf_name]
        self._trials_no_ascend = kwargs['trials_no_ascend']

    def save(self):
        return {
            'best_%s' % self._iter_name: self._best_iter, 'best_%s' % self._perf_name: self._best_perf,
            'trials_no_ascend': self._trials_no_ascend
        }

    @property
    def best_iter(self):
        return self._best_iter
    
    @property
    def best_perf(self):
        return self._best_perf

    @property
    def early_stop(self):
        if (self._early_stop_trials > 0) and (self._trials_no_ascend >= self._early_stop_trials):
            return True
        else:
            return False

    def update(self, iter_index, new_perf):
        # Better
        if self._best_perf is None or self._lmd_ascend_perf(new_perf, self._best_perf):
            # Ret current best iter as 'last_best_iter'
            ret = self._best_iter
            # Update
            self._best_iter = iter_index
            self._best_perf = new_perf
            self._trials_no_ascend = 0
            return ret
        # Update trials
        else:
            self._trials_no_ascend += 1
            return -1


class FreqCounter(ResumableMeter):
    """
    Handling frequency.
    """
    def __init__(self, freq, iter_fetcher=None):
        # Config
        self._freq = freq
        # 1. Values
        self._iter_last_triggered = -1
        self._status = False
        # 2. Iter fetcher
        if iter_fetcher is not None:
            self._iter_fetcher = iter_fetcher
            setattr(self, 'iter_fetcher', iter_fetcher)

    def load(self, **kwargs):
        self._iter_last_triggered, self._status = kwargs['iter_last_triggered'], kwargs['status']

    def save(self):
        return {'iter_last_triggered': self._iter_last_triggered, 'status': self._status}

    @property
    def status(self):
        return self._status

    def check(self, iteration=None, virtual=False):
        if self._freq <= 0: return False
        # Get iteration.
        if hasattr(self, '_iter_fetcher'):
            assert iteration is None
            iteration = self._iter_fetcher()
        # Update
        if (iteration+1)//self._freq > (self._iter_last_triggered+1)//self._freq:
            if virtual: return True
            self._iter_last_triggered = iteration
            self._status = True
        else:
            if virtual: return False
            self._status = False
        # Return
        return self._status


class TriggerLambda(ResumableMeter):
    """
    Triggered by a function.
    """
    def __init__(self, lmd_trigger, n_fetcher=None):
        # Config
        self._lmd_trigger = lmd_trigger
        # 1. Fetcher
        if n_fetcher is not None:
            self._n_fetcher = n_fetcher
            setattr(self, 'n_fetcher', n_fetcher)
        # 2. Status
        self._first = None

    def load(self, **kwargs):
        self._first = kwargs['first']

    def save(self):
        return {'first': self._first}
        
    @property
    def first(self):
        return self._first

    def check(self, n=None):
        if hasattr(self, '_n_fetcher'):
            assert n is None
            n = self._n_fetcher()
        # Check
        ret = self._lmd_trigger(n)
        if ret:
            if self._first is None: self._first = True
            else: self._first = False
        # Return
        return ret


class TriggerPeriod(ResumableMeter):
    """
    Trigger using period:
        For the example 'period=10, area=3', then 0,1,2 (valid), 3,4,5,6,7,8,9 (invalid).
        For the example 'period=10, area=-3', then 0,1,2,3,4,5,6 (invalid), 7,8,9 (valid).
    """
    def __init__(self, period, area):
        assert period >= 0 and period >= area
        # Get lambda & init
        self._lmd_trigger = (lambda n: n < area) if area >= 0 else (lambda n: n >= period + area)
        # Configs
        self._period = period
        # Data
        self._count = 0
        self._n_valid, self._n_invalid = 0, 0

    @property
    def n_valid(self):
        return self._n_valid

    @property
    def n_invalid(self):
        return self._n_invalid

    def load(self, **kwargs):
        self._count = kwargs['count']
        self._n_valid = kwargs['n_valid']
        self._n_invalid = kwargs['n_invalid']

    def save(self):
        return {'count': self._count, 'n_valid': self._n_valid, 'n_invalid': self._n_invalid}

    def check(self):
        # 1. Get return
        ret = self._lmd_trigger(self._count)
        # 2. Update counts
        if self._period != 0:
            self._count = (self._count + 1) % self._period
        if ret: self._n_valid += 1
        else: self._n_invalid += 1
        # Return
        return ret


class Pbar(tqdm, ResumableMeter):
    """
    Progress bar.
    """
    def __init__(self, *args, **kwargs):
        super(Pbar, self).__init__(*args, **kwargs)
        # Register upon exit
        atexit.register(lambda: self.close())

    def load(self, **kwargs):
        self.start_t = time.time() - kwargs['elapsed']
        self.n = kwargs['n']
        self.last_print_n = kwargs['last_print_n']

    def save(self):
        return {'elapsed': self.format_dict['elapsed'], 'n': self.n, 'last_print_n': self.last_print_n}


# ----------------------------------------------------------------------------------------------------------------------
# Exponential Moving Average
# ----------------------------------------------------------------------------------------------------------------------

class EMA(ResumableMeter):
    """
    Exponential Moving Average.
    """
    def __init__(self, beta, init=None):
        super(EMA, self).__init__()
        # Config
        self._beta = beta
        # Data
        self._stale = init

    def load(self, **kwargs):
        self._stale = kwargs['avg']

    def save(self):
        return {'avg': self._stale}

    @property
    def avg(self):
        return self._stale

    def update_average(self, new):
        # Update stale
        if new is not None:
            self._stale = new if self._stale is None else \
                self._beta * self._stale + (1.0 - self._beta) * new
        # Return
        return self._stale


class EMAPyTorchModel(ResumableMeter):
    """
    Exponential Moving Average for PyTorch Model.
    """
    def __init__(self, beta, model, **kwargs):
        # Config
        self._beta = beta
        # 1. Data
        self._model, self._initialized = model, False
        # 2. Init
        if 'init' in kwargs.keys():
            self._model.load_state_dict(kwargs['init'].state_dict())
            self._initialized = True

    def load(self, **kwargs):
        self._model.load_state_dict(kwargs['avg_state_dict'])
        self._initialized = kwargs['initialized']

    def save(self):
        return {
            'avg_state_dict': self._model.state_dict(),
            'initialized': self._initialized
        }

    @property
    def initialized(self):
        return self._initialized

    @property
    def avg(self):
        return self._model

    def update_average(self, new):
        # Update stale
        if new is not None:
            # 1. Init
            if not self._initialized:
                self._model.load_state_dict(new.state_dict())
                self._initialized = True
            # 2. Moving average
            else:
                for stale_param, new_param in zip(self._model.parameters(), new.parameters()):
                    stale_param.data = self._beta * stale_param.data + (1.0 - self._beta) * new_param.data
        # Return
        return self._model


# ----------------------------------------------------------------------------------------------------------------------
# Timers
# ----------------------------------------------------------------------------------------------------------------------

class StopWatch(object):
    """
    Timer for recording durations.
    """
    def __init__(self):
        # Statistics - current
        self._stat = 'off'
        self._cur_duration = 0.0
        # Statistics - total
        self._total_duration = 0.0

    @property
    def stat(self):
        return self._stat

    def resume(self):
        # Record start time, switch to 'on'
        self._cur_duration = time.time()
        self._stat = 'on'

    def pause(self):
        if self._stat == 'off': return
        # Get current duration, switch to 'off'
        self._cur_duration = time.time() - self._cur_duration
        self._stat = 'off'
        # Update total duration
        self._total_duration += self._cur_duration

    def get_duration_and_reset(self):
        result = self._total_duration
        self._total_duration = 0.0
        return result


class _TimersManager(object):
    """
    Context manager for timers.
    """
    def __init__(self, timers, cache):
        # Config
        self._timers = timers
        self._cache = cache

    def __enter__(self):
        if self._cache is None: return
        # Activate
        for k in self._cache['on']:
            self._timers[k].resume()
        for k in self._cache['off']:
            self._timers[k].pause()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cache is None: return
        # Restore
        for k in self._cache['on']:
            self._timers[k].pause()
        for k in self._cache['off']:
            self._timers[k].resume()


class TimersController(object):
    """
    Controller for a bunch of timers.
    """
    def __init__(self, **kwargs):
        # Members
        self._timers = {}
        # Set timers
        for k, v in kwargs.items():
            self[k] = v

    def __contains__(self, key):
        return key in self._timers.keys()

    def __setitem__(self, key, val):
        assert key not in self._timers.keys() and isinstance(val, StopWatch) and val.stat == 'off'
        # Save
        self._timers[key] = val

    def __getitem__(self, key):
        return self._timers[key]

    def __call__(self, *args, **kwargs):
        # Calculate cache
        if not chk_d(kwargs, 'void'):
            # 1. On
            on = list(filter(lambda _k: _k in self._timers, args))
            for k in on: assert self._timers[k].stat == 'off'
            # 2. Off
            off = [k for k in filter(lambda _k: self._timers[_k].stat == 'on', self._timers.keys())]
            # Result
            cache = {'on': on, 'off': off}
        else:
            cache = None
        # Return
        return _TimersManager(self._timers, cache=cache)


########################################################################################################################
# Metrics
########################################################################################################################

def basic_stat(data, axis=-1, **kwargs):
    """
    :param data: np.array.
    :param axis:
    :param kwargs:
        - conf_percent:
        - minmax:
    :return:
    """
    # 1. Calculate avg & std.
    results = {'avg': data.mean(axis=axis), 'std': data.std(ddof=1, axis=axis)}
    # 2. Calculate CI.
    if 'conf_percent' in kwargs.keys():
        # (1) Get 'n'
        n = {90: 1.645, 95: 1.96, 99: 2.576}[kwargs['conf_percent']]
        # (2) Get interval
        results['interval'] = n * results['std']
    # 3. Min & max
    if 'minmax' in kwargs.keys() and kwargs['minmax']:
        results.update({'min': data.min(), 'max': data.max()})
    # Return
    return results


def mean_accuracy(global_gt, global_pred, num_classes=None):
    """
    Mean Accuracy for classification.
    :param global_gt: (N, )
    :param global_pred: (N, )
    :param num_classes: Int. Provided for avoiding inference.
    :return:
    """
    # Infer num_classes
    if num_classes is None: num_classes = len(set(global_gt))
    # (1) Init result
    mean_acc = 0
    classes_acc = []
    # (2) Process each class
    for i in range(num_classes):
        cur_indices = np.where(global_gt == i)[0]
        # For current class
        cur_acc = accuracy_score(global_gt[cur_indices], global_pred[cur_indices])
        # Add
        mean_acc += cur_acc
        classes_acc.append(cur_acc)
    # (3) Get result
    mean_acc = mean_acc / num_classes
    # Return
    return mean_acc, np.array(classes_acc)


def api_empty_cache(func):
    @wraps(func)
    def _wrapped(*args, **kwargs):
        torch.cuda.empty_cache()
        ret = func(*args, **kwargs)
        torch.cuda.empty_cache()
        return ret

    return _wrapped


def api_eval_torch(func):
    @wraps(func)
    def _api(*args, **kwargs):
        with TempKwargsManager(args[0], **kwargs):
            torch.cuda.empty_cache()
            ret = func(*args)
            torch.cuda.empty_cache()
            return ret

    return _api


# ----------------------------------------------------------------------------------------------------------------------
# Evaluating Disentanglement: Quantatitive metrics.
# ----------------------------------------------------------------------------------------------------------------------

def _accum_logsumexp(result, x, dim):
    if result is None: return torch.logsumexp(x, dim=dim)
    return torch.logsumexp(torch.cat([result.unsqueeze(dim), x], dim=dim), dim=dim)


def _accum_add(result, x):
    if result is None: return x
    return result + x


def _accum_cat(result, x, dim=0):
    if result is None: return x
    return torch.cat([result, x], dim=dim)


class DisenEvaluatorQuant(object):
    """
    Disentanglement evaluator for quantatitive meters.
    """
    def __init__(self, func_encoder, **kwargs):
        """
        :param func_encoder: A mapping which takes x as input, and outputs
            - z_params: mu, logvar. (batch, nz)
        """
        # Config.
        self._kwargs = kwargs
        # Setup encoder.
        self._Enc = func_encoder

    """ OVERRIDABLE """
    def _deploy_batch_data(self, batch_data):
        """
            The method specifies that how to deploy the data, given a batch_data from BatchSlicerLenObj.
        :param batch_data:
        :return: (batch, C, H, W)
        """
        if isinstance(batch_data, np.ndarray): batch_data = torch.from_numpy(batch_data)
        return batch_data.to(self._kwargs['device'])

    @api_eval_torch
    def estimate_z_entropies(self, x_dataset, pbar_desc="Estimating H_q(Z), H_q(Z_j), H_q(Z_j|X) & H_cross[q(Z_j|X)||p(Z_j)]", **kwargs):
        """
        :param x_dataset: Ports:
                - __len__
                - __getitem__(index)
        :param pbar_desc:
        ----------------------------------------------------------------------------------------------------------------
        :return:
            - H_q(Z) =                                                                      (1, )
                * -1/Z sum_z log q(z) = -1/Z sum_z log{ 1/N sum_n q(z|n) }.
            - H_q(Z_j) =                                                                    (nz, )
                * -1/Z sum_z log q(z_j) = -1/Z sum_z { log.sum.exp{along_n}{ sum_j log q(z_j|n) } - log N }.
            - H_q(Z_j|X) by analytically computing.                                         (nz, )
            - H_cross[ q(Z_j|X) || p(Z_j) ] by analytically computing.                      (nz, )
        ----------------------------------------------------------------------------------------------------------------
        """
        ################################################################################################################
        # 1. Calculate log densities & conditional entropies.
        # (1) Marginal density.                 log_q(z_j).                         (total_n_samples, nz)
        # (2) Aggregated density.               log_q(z).                           (total_n_samples, )
        # (3) Conditional marginal entropy.     H_q(Z_j|X).                         (nz, )
        # (4) Conditional cross entropy.        H_cross[ q(Z_j|X) || p(Z_j) ].      (nz, )
        ################################################################################################################
        # (1) Init results.
        log_p_z_marginal, log_p_z, ent_z_marginal_params, ent_cross_z_marginal_params_with_prior = None, None, None, None
        # (2) Calculate from each batch.
        x_dataloader, total_n_samples = BatchSlicerLenObj(x_dataset, self._kwargs['ent_z_batch_size'], self._kwargs['ent_z_max_counts']), 0
        pbar = tqdm(total=len(x_dataloader)**2, desc=pbar_desc) if pbar_desc is not None else None
        for batch_x_index, batch_x in enumerate(x_dataloader):
            # Sampling z. (batch_z, nz)
            batch_z = reparameterize(*self._Enc(self._deploy_batch_data(batch_x)), n_samples=self._kwargs['ent_z_n_sampling'], squeeze=True)
            total_n_samples += len(batch_z)
            # ----------------------------------------------------------------------------------------------------------
            # Calculate
            #   - Marginal density.             log p(curz_j).              (batch_z, nz)
            #   - Aggregated density.           log p(curz).                (batch_z, )
            # ----------------------------------------------------------------------------------------------------------
            # (1) Init results.
            log_p_curz_marginal, log_p_curz = None, None
            # (2) Calculate from each batch.
            p_dataloader = BatchSlicerLenObj(x_dataset, self._kwargs['ent_z_batch_size'], self._kwargs['ent_z_max_counts'])
            for batch_params_index, batch_params_x in enumerate(p_dataloader):
                # Get z_params. (batch_params, nz)
                z_params = self._Enc(self._deploy_batch_data(batch_params_x))
                ########################################################################################################
                # Calculate for non-conditional by ESTIMATING
                #   - Marginal density.         log p(curz_j|x).            (batch_z, batch_params, nz)
                #   - Aggregated density.       log p(curz|x).              (batch_z, batch_params)
                ########################################################################################################
                log_p_curz_marginal_curparams = gaussian_log_density_marginal(batch_z, z_params, mesh=True).cpu()
                log_p_curz_curparams = log_p_curz_marginal_curparams.sum(dim=2).cpu()
                """ Accumulate to results. """
                log_p_curz_marginal = _accum_logsumexp(log_p_curz_marginal, log_p_curz_marginal_curparams, dim=1)
                log_p_curz = _accum_logsumexp(log_p_curz, log_p_curz_curparams, dim=1)
                ########################################################################################################
                # Calculate for conditional by ANALYTIC
                #   - Marginal entropy.             H_q(Z_j|x).                         (batch_params, nz)
                #   - Conditional cross entropy.    H_cross[ q(Z_j|x) || p(Z_j) ].      (batch_params, nz)
                ########################################################################################################
                if batch_x_index == 0:
                    ent_z_marginal_curparams = gaussian_cross_entropy_marginal(z_params, 'entropy').cpu()
                    ent_cross_z_marginal_curparams_with_prior = gaussian_cross_entropy_marginal(z_params, None).cpu()
                    """ Accumulate to results """
                    ent_z_marginal_params = _accum_add(ent_z_marginal_params, ent_z_marginal_curparams.sum(dim=0))
                    ent_cross_z_marginal_params_with_prior = _accum_add(ent_cross_z_marginal_params_with_prior, ent_cross_z_marginal_curparams_with_prior.sum(dim=0))
                """ Show progress. """
                if pbar is not None: pbar.update(1)
            log_p_curz_marginal, log_p_curz = map(lambda _x: _x - math.log(p_dataloader.max_counts), [log_p_curz_marginal, log_p_curz])
            # ----------------------------------------------------------------------------------------------------------
            # Accumulate to results.
            # ----------------------------------------------------------------------------------------------------------
            log_p_z_marginal = _accum_add(log_p_z_marginal, log_p_curz_marginal.sum(0))
            log_p_z = _accum_add(log_p_z, log_p_curz.sum(0))
            # Normalizing conditional entropies.
            if batch_x_index == 0:
                ent_z_marginal_params /= p_dataloader.max_counts
                ent_cross_z_marginal_params_with_prior /= p_dataloader.max_counts
        if pbar is not None: pbar.close()
        """ Saving """
        ret_entropies = {
            'ent_cond_z_marginal': ent_z_marginal_params,
            'ent_cond_cross_z_marginal_with_prior': ent_cross_z_marginal_params_with_prior}
        ################################################################################################################
        # 2. Calculate unconditional entropies.
        # (1) Marginal entropy.                 H_q(Z_j).                           (nz, )
        # (2) Aggregated entropy.               H_q(Z).                             (1, )
        ################################################################################################################
        ent_z_marginal = - log_p_z_marginal / total_n_samples
        ent_z = - log_p_z / total_n_samples
        """ Saving """
        ret_entropies.update({'ent_z_marginal': ent_z_marginal, 'ent_z': ent_z})
        # Return
        return ret_entropies

    @api_eval_torch
    def estimate_elbo(self, dataset_generator, z_entropies=None, **kwargs):
        # Get H_q(Z), H_q(Z_j), H_q(Z_j|X), H_cross[q(Z_j|X)||p(Z_j)].
        if z_entropies is None: z_entropies = self.estimate_z_entropies(dataset_generator.subset(), "Estimating ELBO @ Z entropies")
        # --------------------------------------------------------------------------------------------------------------
        # 1. Original KL. (1, )
        # E_{p(x)} KL[ q(z|x) || p(z) ] = H_cross[ q(Z|X), p(Z) ] - H_q(Z|X)
        # --------------------------------------------------------------------------------------------------------------
        orig_kl = (z_entropies['ent_cond_cross_z_marginal_with_prior'] - z_entropies['ent_cond_z_marginal']).sum()
        # --------------------------------------------------------------------------------------------------------------
        # 2.1 MI in KL divergence. (1, )
        # I_q(X;Z) = E_{p(x)} KL[ q(z|x) || q(z) ] = H_q(Z) - H_q(Z|X).
        # --------------------------------------------------------------------------------------------------------------
        mi = z_entropies['ent_z'] - z_entropies['ent_cond_z_marginal'].sum()
        # --------------------------------------------------------------------------------------------------------------
        # 2.2 Total Correlation. (1, )
        # KL[ q(z) || prod_j q(z_j) ] ~= sum_j H_q(Z_j) - H_q(Z)
        # --------------------------------------------------------------------------------------------------------------
        tc = z_entropies['ent_z_marginal'].sum() - z_entropies['ent_z']
        # --------------------------------------------------------------------------------------------------------------
        # 2.3 Dimension-wise KL. (1, )
        # sum_j KL[ q(z_j) || p(z_j) ] ~= sum_j { E_{p(x)} H_cross[ q(Z_j|x), p(Z_j) ] - H_q(Z_j) }
        # --------------------------------------------------------------------------------------------------------------
        dimwise_kl = (z_entropies['ent_cond_cross_z_marginal_with_prior'] - z_entropies['ent_z_marginal']).sum()
        # Return
        return {'orig_kl': orig_kl.item(), 'mi': mi.item(), 'tc': tc.item(), 'dimwise_kl': dimwise_kl.item()}

    @api_eval_torch
    def estimate_mig(self, dataset_generator, ent_z_marginal=None, **kwargs):
        """
        :param dataset_generator: Ports:
            - subset(factors=None)                              Returns the full dataset with '__len__' & '__getitem__(index)' ports.
                * factors = { factor1_name: [factor1_value1, factor1_value2, ..., ], ... }
                * if not given factors, return the full dataset.
            - factors:                                          Returns { factor1_name: [factor1_value1, factor1_value2, ..., ], ... } (ordered).
            - n_factors_values:                                 Returns [n_f1_values, n_f2_values, ...]
        :param ent_z_marginal:
        :return:
        """
        ################################################################################################################
        # Estimating I_q(Z_j;V_k) = H_q(Z_j) - H_q(Z_j|V_k). (num_factors, nz)
        ################################################################################################################
        # 1. Calculate H_q(Z_j). (nz, )
        if ent_z_marginal is None: ent_z_marginal = self.estimate_z_entropies(dataset_generator.subset(), "Estimating MIG @ H_q(Z_j)")['ent_z_marginal']
        # 2. Calculate H_q(Z_j|V_k). (num_factors, nz)
        ent_z_marginal_given_factors = []
        # (1) Process each factor
        pbar = tqdm(total=sum(dataset_generator.n_factors_values))
        for factor_name, fvs in dataset_generator.factors.items():
            ent_z_marginal_given_f = []
            # 1> process each value.
            for fv in fvs:
                pbar.set_description("Estimating MIG @ H_q(Z_j|%s=%d)" % (factor_name, fv))
                ent_z_marginal_given_f.append(
                    self.estimate_z_entropies(dataset_generator.subset({factor_name: [fv]}), None)['ent_z_marginal'].unsqueeze(0))
                pbar.update(1)
            # 2> Get result.
            ent_z_marginal_given_f = torch.cat(ent_z_marginal_given_f, dim=0).mean(0, keepdim=True)
            """ Saving """
            ent_z_marginal_given_factors.append(ent_z_marginal_given_f)
        pbar.close()
        # (2) Get results
        ent_z_marginal_given_factors = torch.cat(ent_z_marginal_given_factors)
        # 3. Get MI. (num_factors, nz)
        mi_z_marginal_with_factors = ent_z_marginal.unsqueeze(0) - ent_z_marginal_given_factors
        ################################################################################################################
        # Estimating MIG. (num_factors, dims)
        ################################################################################################################
        # 1. Get top2 max dims for each factor. (num_factors, 2)
        top2_mi = torch.topk(mi_z_marginal_with_factors, k=2, dim=1)
        _1st_value, _2nd_value = top2_mi.values[:, 0], top2_mi.values[:, 1]
        # 2. Get normalizer H(factor). (num_factors, )
        constant = torch.log(torch.tensor(dataset_generator.n_factors_values, dtype=torch.float32, device=top2_mi.values.device))
        # Get MIG. (num_factors, )
        migs = (_1st_value - _2nd_value) / constant
        """ Saving """
        scores = {'_overall': migs.mean().item()}
        for k, v, dim in zip(dataset_generator.factors.keys(), migs, top2_mi.indices[:, 0]):
            assert k not in scores.keys()
            scores['%s@%d' % (k, dim.item())] = v.item()
        # Return
        return scores

    @api_eval_torch
    def estimate_elbo_and_mig(self, dataset_generator, **kwargs):
        # 1. Get z entropies.
        z_entropies = self.estimate_z_entropies(dataset_generator.subset(), "Estimating Z entropies")
        # 2. Get ELBO & MIG.
        ret_elbo = self.estimate_elbo(None, z_entropies)
        ret_mig = self.estimate_mig(dataset_generator, z_entropies['ent_z_marginal'])
        # return
        return {'elbo': ret_elbo, 'mig': ret_mig}


def _discretize(z, interval_start, interval_end, nbins):
    """
    :param z: (batch, nz)
    :param interval_start: (nz, )
    :param interval_end: (nz, )
    :param nbins:
    :return: (batch, nz, nbins).
    """
    assert interval_start.gt(interval_end).sum().item() == 0
    # 1. Get indices. (batch, nz)
    interval_start, interval_end = map(lambda _x: _x.unsqueeze(0), [interval_start, interval_end])
    interval = interval_end - interval_start
    # Get indices: handling zero interval case.
    indices = (((z-interval_start) / interval)*nbins).clamp(0, nbins-0.5).floor().to(torch.int64)
    indices = torch.where(interval.gt(0.0), indices, torch.zeros_like(indices))
    # 2. To (batch, nz, nbins).
    ret = F.one_hot(indices, nbins)
    # Return
    return ret


class DisenMetricMIG(object):
    """
    Disentanglement quantitative metric: MIG, JEMMIG, MIG-sup, Modularity Score, DCIMIG
    """
    def __init__(self, func_encoder, **kwargs):
        """
        :param func_encoder: A mapping which takes x as input, and outputs z.
        :param kwargs:
        """
        # Config.
        self._kwargs = kwargs
        # Setup encoder.
        self._Enc = func_encoder

    """ OVERRIDABLE """
    def _deploy_batch_data(self, batch_data):
        return batch_data.to(self._kwargs['device'])

    ####################################################################################################################
    # Metrics calculations
    ####################################################################################################################

    def _calc_mig_and_jemmig(self, mi_matrix, ent_joint_z_marginals_factors, ent_f_marginals, factor_names):
        """
        :param mi_matrix: I(Z_j;V_k). (num_factors, nz).
        :param ent_joint_z_marginals_factors: H(Z_j, V_k). (num_factors, nz).
        :param ent_f_marginals: (num_factors, )
        :param factor_names: List.
        :return:
            mig: { '_overall': score, 'f@dim': score, ... }
            jemmig: The same structure as above.
        """
        # --------------------------------------------------------------------------------------------------------------
        # Get MI gaps. (num_factors, )
        # --------------------------------------------------------------------------------------------------------------
        # 1. Get top2 max dims for each factors. (num_factors, 2)
        top2_mi = torch.topk(mi_matrix, k=2, dim=1)
        _1st_value, _2nd_value, max_dim_indices = top2_mi.values[:, 0], top2_mi.values[:, 1], top2_mi.indices[:, 0]
        # 2. Get MI gaps. (num_factors, )
        mi_gaps = _1st_value - _2nd_value
        # --------------------------------------------------------------------------------------------------------------
        # METRIC: MIG.
        # --------------------------------------------------------------------------------------------------------------
        # 1. Get MIG. (num_factors, )
        migs = mi_gaps / mi_matrix.sum(dim=1)
        """ Get scores """
        mig_scores = {'_overall': migs.mean().item()}
        for k, v, dim in zip(factor_names, migs, max_dim_indices):
            mig_scores['%s@%d' % (k, dim.item())] = v.item()
        # --------------------------------------------------------------------------------------------------------------
        # METRIC: JEMMIG
        # --------------------------------------------------------------------------------------------------------------
        # 1. Get JEMMIG. (num_factors, )
        mask = F.one_hot(max_dim_indices, mi_matrix.size(1))
        jemmigs = ((ent_joint_z_marginals_factors*mask).sum(dim=1) - mi_gaps) / (ent_f_marginals + math.log(self._kwargs['ent_z_nbins']))
        """ Get scores """
        jemmig_scores = {'_overall': jemmigs.mean().item()}
        for k, v, dim in zip(factor_names, jemmigs, max_dim_indices):
            jemmig_scores['%s@%d' % (k, dim.item())] = v.item()
        # --------------------------------------------------------------------------------------------------------------
        # Return
        return mig_scores, jemmig_scores

    @staticmethod
    def _calc_migsup(mi_matrix, ent_f_marginals, factor_names):
        """
        :param mi_matrix: I(Z_j;V_k). (num_factors, nz)
        :param ent_f_marginals: (num_factors, )
        :param factor_names:
        :return:
            mig: { '_overall': score, 'dim@f': score, ... }
        """
        # --------------------------------------------------------------------------------------------------------------
        # Get MI gaps. (nz, )
        # --------------------------------------------------------------------------------------------------------------
        # 1. Get top2 max factors for each dims. (2, nz)
        top2_mi = torch.topk(mi_matrix / ent_f_marginals.unsqueeze(1), k=2, dim=0)
        _1st_value, _2nd_value, max_f_indices = top2_mi.values[0, :], top2_mi.values[1, :], top2_mi.indices[0, :]
        # 2. Get MI gaps. (nz, )
        mi_gaps = _1st_value - _2nd_value
        # --------------------------------------------------------------------------------------------------------------
        # METRIC: MIG-sup
        # --------------------------------------------------------------------------------------------------------------
        """ Get scores """
        scores = {'_overall': mi_gaps.mean().item()}
        for dim, v in enumerate(mi_gaps):
            scores['%d@%s' % (dim, factor_names[max_f_indices[dim].item()])] = v.item()
        # Return
        return scores

    @staticmethod
    def _calc_modularity_score(mi_matrix, factor_names):
        """
        :param mi_matrix: I(Z_j;V_k). (num_factors, nz)
        :param factor_names:
        :return:
            mig: { '_overall': score, 'dim@f': score, ... }
        """
        # --------------------------------------------------------------------------------------------------------------
        # Get mask. (num_factors, nz)
        # --------------------------------------------------------------------------------------------------------------
        num_factors = mi_matrix.size(0)
        # 1. Get indices. (nz, )
        indices = torch.topk(mi_matrix, k=1, dim=0).indices[0]
        # 2. Get mask. (num_factors, nz)
        mask = F.one_hot(indices, num_factors).T
        # 3. Get value. (nz, )
        mi_matrix_squared = mi_matrix**2
        other_factors, max_factor = ((1.0 - mask)*mi_matrix_squared).sum(dim=0), (mask*mi_matrix_squared).sum(dim=0)*(num_factors-1)
        values = 1.0 - other_factors/max_factor
        # --------------------------------------------------------------------------------------------------------------
        # METRIC: modularity score
        # --------------------------------------------------------------------------------------------------------------
        """ Get scores """
        scores = {'_overall': values.mean().item()}
        for dim, v in enumerate(values):
            scores['%d@%s' % (dim, factor_names[indices[dim].item()])] = v.item()
        # Return
        return scores

    @staticmethod
    def _calc_dcimig(mi_matrix, ent_f_marginals):
        """
        :param mi_matrix: I(Z_j;V_k). (num_factors, nz)
        :param ent_f_marginals: (num_factors, )
        :return A single score scalar.
        """
        # --------------------------------------------------------------------------------------------------------------
        # 1. Get MI gaps. (nz, )
        # --------------------------------------------------------------------------------------------------------------
        top2_mi = torch.topk(mi_matrix, k=2, dim=0)
        _1st_value, _2nd_value, max_f_indices = top2_mi.values[0, :], top2_mi.values[1, :], top2_mi.indices[0, :]
        mi_gaps = _1st_value - _2nd_value
        # --------------------------------------------------------------------------------------------------------------
        # 2. Get S. (num_factors, )
        # --------------------------------------------------------------------------------------------------------------
        # 1. Get gap_matrix. (num_factors, nz)
        gap_matrix = F.one_hot(max_f_indices, mi_matrix.size(0)).T * mi_gaps.unsqueeze(0)
        # 2. Get S. (num_factors, )
        s = gap_matrix.max(dim=1).values
        # --------------------------------------------------------------------------------------------------------------
        # METRIC: DCIMIG
        # --------------------------------------------------------------------------------------------------------------
        """ Get score """
        score = (s.sum() / ent_f_marginals.sum()).item()
        # Return
        return score

    ####################################################################################################################
    # APIs
    ####################################################################################################################

    @api_eval_torch
    @api_empty_cache
    @torch.no_grad()
    def estimate_meanstd(self, x_dataset, pbar_desc="Estimating mean & std for z_j", **kwargs):
        """
        :param x_dataset: Wrapped by a DataLoader, batch_data should be torch.Tensor. Ports:
            - __len__
            - __getitem__(index)
        :param pbar_desc:
        :param kwargs:
        :return: Mean & std for each z_j. (nz, ) & (nz, )
        """
        x_dataloader = DataLoader(x_dataset, batch_size=self._kwargs['ent_z_batch_size'], shuffle=True)
        ################################################################################################################
        # Get z. (max_counts, nz)
        ################################################################################################################
        # 1. Init results.
        codes = []
        # 2. Calculate from each batch.
        total_n_samples, pbar = 0, (tqdm(total=self._kwargs['ent_z_max_counts'], desc=pbar_desc) if pbar_desc is not None else None)
        for batch_index, batch_x in enumerate(x_dataloader):
            # ----------------------------------------------------------------------------------------------------------
            # Get z. (batch, nz)
            batch_z = self._Enc(self._deploy_batch_data(batch_x))
            batch_size = min(len(batch_z), self._kwargs['ent_z_max_counts']-total_n_samples)
            batch_z = batch_z[:batch_size]
            """ Accumulate """
            codes.append(batch_z.cpu())
            # Update
            total_n_samples += batch_size
            if pbar is not None: pbar.update(batch_size)
            # ----------------------------------------------------------------------------------------------------------
            if total_n_samples == self._kwargs['ent_z_max_counts']: break
        if pbar is not None: pbar.close()
        # 3. Get results.
        codes = torch.cat(codes)
        ################################################################################################################
        # Get mean & std. (nz, ) & (nz, )
        ################################################################################################################
        mean, std = codes.mean(dim=0), codes.std(dim=0, unbiased=False)
        # Return
        return mean, std

    @api_eval_torch
    @api_empty_cache
    @torch.no_grad()
    def estimate_marginal_entropies(self, x_dataset, pbar_desc="Estimating H(Z_j)", **kwargs):
        """
        :param x_dataset: Wrapped by a DataLoader, batch_data should be torch.Tensor. Ports:
            - __len__
            - __getitem__(index)
        :param pbar_desc:
        :param kwargs:
        :return: Entropy of discretized q(z_j). (nz, )
        """
        x_dataloader = DataLoader(x_dataset, batch_size=self._kwargs['ent_z_batch_size'], shuffle=True)
        ################################################################################################################
        # Get discretized probability. (nz, nbins)
        ################################################################################################################
        # 1. Init results. (nz, nbins)
        prob = None
        # 2. Calculate from each batch.
        total_n_samples, pbar = 0, (tqdm(total=self._kwargs['ent_z_max_counts'], desc=pbar_desc) if pbar_desc is not None else None)
        for batch_index, batch_x in enumerate(x_dataloader):
            # ----------------------------------------------------------------------------------------------------------
            # Get z. (batch, nz)
            batch_z = self._Enc(self._deploy_batch_data(batch_x))
            batch_size = min(len(batch_z), self._kwargs['ent_z_max_counts']-total_n_samples)
            batch_z = batch_z[:batch_size]
            """ Accumulate """
            cur_prob = _discretize(batch_z, self._kwargs['ent_z_interval_start'], self._kwargs['ent_z_interval_end'], self._kwargs['ent_z_nbins']).sum(dim=0).to(torch.float32)
            prob = cur_prob if prob is None else (prob + cur_prob)
            # Update
            total_n_samples += batch_size
            if pbar is not None: pbar.update(batch_size)
            # ----------------------------------------------------------------------------------------------------------
            if total_n_samples == self._kwargs['ent_z_max_counts']: break
        if pbar is not None: pbar.close()
        # 3. Get results.
        prob = prob / prob.sum(dim=1, keepdim=True)
        ################################################################################################################
        # Get discretized entropy. (nz, )
        ################################################################################################################
        ent = (-prob*torch.log(prob+1e-8)).sum(dim=1).cpu()
        # Return
        return ent

    @api_eval_torch
    @torch.no_grad()
    def __call__(self, dataset_generator, **kwargs):
        """
        :param dataset_generator: Ports:
            - subset(factors=None)                              Returns datasets specified in the above 'estimate_entropy' method.
                * factors = { factor1_name: [factor1_value1, factor1_value2, ..., ], ... }
                * if not given factors, return the full dataset.
            - factors:                                          Returns { factor1_name: [factor1_value1, factor1_value2, ..., ], ... } (ordered).
            - n_factors_values:                                 Returns [n_f1_values, n_f2_values, ...]
        :param ent_z_marginal:
        :return:
        """
        ################################################################################################################
        # Determining intervals for each z.
        ################################################################################################################
        # 1. Get mean & std. (nz, ) & (nz, )
        mean, std = map(lambda _x: _x.to(self._kwargs['device']), self.estimate_meanstd(dataset_generator.subset()))
        # 2. Get interval=(mean-3std, mean+3std). (nz, ) & (nz, )
        interval_kwargs = {'ent_z_interval_start': mean-3*std, 'ent_z_interval_end': mean+3*std}
        ################################################################################################################
        # Estimating I(Z_j;V_k) = H(Z_j) - H(Z_j|V_k). (num_factors, nz)
        ################################################################################################################
        # 1. Calculate H(Z_j). (nz, )
        ent_z_marginal = self.estimate_marginal_entropies(dataset_generator.subset(), **interval_kwargs)
        # 2. Calculate H(Z_j|V_k). (num_factors, nz)
        ent_z_marginal_given_factors = []
        # (1) Process each factor
        pbar = tqdm(total=sum(dataset_generator.n_factors_values))
        for factor_name, factor_values in dataset_generator.factors.items():
            ent_z_marginal_given_f = []
            # 1> process each value.
            for fv in factor_values:
                pbar.set_description("Estimating H(Z_j|%s=%d)" % (factor_name, fv))
                ent_z_marginal_given_f.append(
                    self.estimate_marginal_entropies(dataset_generator.subset({factor_name: [fv]}), None, **interval_kwargs).unsqueeze(0))
                pbar.update(1)
            # 2> Get result.
            ent_z_marginal_given_f = torch.cat(ent_z_marginal_given_f, dim=0).mean(0, keepdim=True)
            """ Saving """
            ent_z_marginal_given_factors.append(ent_z_marginal_given_f)
        pbar.close()
        # (2) Get results
        ent_z_marginal_given_factors = torch.cat(ent_z_marginal_given_factors)
        # 3. Get MI. (num_factors, nz)
        mi_matrix = ent_z_marginal.unsqueeze(0) - ent_z_marginal_given_factors
        ################################################################################################################
        # Estimating METRICS.
        ################################################################################################################
        # Get H(V_k). (num_factors, ) & H(Z_j, V_k). (num_factors, nz)
        ent_f_marginals = torch.tensor(dataset_generator.n_factors_values, dtype=torch.float32).log()
        ent_joint_z_marginals_factors = ent_f_marginals.unsqueeze(1) + ent_z_marginal_given_factors
        """ Get metrics """
        factor_names = list(dataset_generator.factors.keys())
        mig, jemmig = self._calc_mig_and_jemmig(mi_matrix, ent_joint_z_marginals_factors, ent_f_marginals, factor_names)
        migsup = self._calc_migsup(mi_matrix, ent_f_marginals, factor_names)
        modularity_score = self._calc_modularity_score(mi_matrix, factor_names)
        dcimig = self._calc_dcimig(mi_matrix, ent_f_marginals)
        # Return
        return {'mig': mig, 'jemmig': jemmig, 'migsup': migsup, 'modularity_score': modularity_score, 'dcimig': dcimig}


# ----------------------------------------------------------------------------------------------------------------------
# Evaluating Disentanglement: Qualitative latent traversal.
# ----------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def vis_latent_traversal(func_decoder, z, limit=3.0, n_traversals=10):
    """
    :param func_decoder: A mapping which takes z as input, and outputs reconstruction. (batch, C, H, W)
    :param z: (batch, nz)
    :param limit:
    :param n_traversals:
    :return: (batch, nz, 1+n_traversals, C, H, W), where '1' represents for reconstruction.
    """
    bsize, nz = z.size()
    # ------------------------------------------------------------------------------------------------------------------
    # Get reconstruction. (batch, C, H, W)
    # ------------------------------------------------------------------------------------------------------------------
    x_recon = func_decoder(z).cpu()
    # ------------------------------------------------------------------------------------------------------------------
    # Get latent traversal. (batch, nz, n_traversals, C, H, W)
    # ------------------------------------------------------------------------------------------------------------------
    # Interpolation. (n_traversals, )
    interp = torch.arange(-limit, limit+1e-8, step=2.0*limit/(n_traversals-1), device=z.device)
    # 1. Get interp z. (batch, nz, n_traverals, nz)
    z = z.unsqueeze(1).unsqueeze(1).expand(bsize, nz, n_traversals, nz)
    mask_interp = torch.eye(nz, dtype=z.dtype, device=z.device).unsqueeze(1).unsqueeze(0)
    z_interp = z * (1.0-mask_interp) + interp.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * mask_interp
    # 2. Get decoded. (batch*nz,*n_traversals, C, H, W) -> (batch, nz, n_traversals, C, H, W)
    x_traverals = []
    for z in BatchSlicerLenObj(z_interp.reshape(-1, nz), batch_size=16):
        x_traverals.append(func_decoder(z).cpu())
    x_traverals = torch.cat(x_traverals)
    x_traverals = x_traverals.reshape(bsize, nz, n_traversals, *x_traverals.size()[1:])
    # Return
    return torch.cat([x_recon.unsqueeze(1).unsqueeze(1).expand(bsize, nz, 1, *x_recon.size()[1:]), x_traverals], dim=2)


def vis_latent_traversal_given_x(func_encoder, func_decoder, x_real, limit=3.0, n_traversals=10):
    """
    :param func_encoder: A mapping which takes x as input, and outputs z. (batch, nz)
    :param func_decoder: A mapping which takes z as input, and outputs reconstruction. (batch, C, H, W)
    :param x_real: (batch, C, H, W)
    :param limit:
    :param n_traversals:
    :return: (batch, nz, 2+n_traversals, C, H, W), where '2' represents for 'real' & 'reconstruction'.
    """
    # 1. Get (batch, nz, 1+n_traversals, C, H, W)
    z = func_encoder(x_real)
    latent_traversals = vis_latent_traversal(func_decoder, z, limit, n_traversals)
    # 2. Get (batch, nz, 2+n_traversals, C, H, W)
    ret = torch.cat([x_real.unsqueeze(1).unsqueeze(1).expand(z.size(0), z.size(1), 1, *x_real.size()[1:]).cpu(), latent_traversals], dim=2)
    # Return
    return ret


# ----------------------------------------------------------------------------------------------------------------------
# Summarize to .xls.
# ----------------------------------------------------------------------------------------------------------------------

class _TreeNode(object):
    """
    TreeNode for titles.
    """
    def __init__(self, name='root'):
        self._name = name
        self._parent, self._index_in_brother, self._children = None, None, []

    @property
    def name(self):
        return self._name

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    @property
    def row_index(self):
        if self._parent is None: return 0
        return self._parent.row_index + 1

    @property
    def col_index(self):
        # 1. Get parent's col_index.
        parent_col_index = 0 if self._parent is None else self._parent.col_index
        # 2. Get sum of left-brothers' col_expand.
        brothers_col_expands = 0 if self._index_in_brother is None else sum([child.col_expand for child in self._parent._children[:self._index_in_brother]])
        # Return
        return parent_col_index + brothers_col_expands

    @property
    def row_expand(self):
        if self._children: return 1
        # Get root node.
        root = self
        while root.parent is not None: root = root.parent
        # Get result.
        return root.depth_subtree - self.row_index

    @property
    def col_expand(self):
        if not self._children: return 1
        return sum([child.col_expand for child in self._children])

    @property
    def depth_subtree(self):
        if not self._children: return 1
        return max([child.depth_subtree+1 for child in self._children])

    def add_child(self, nodes):
        if isinstance(nodes, str): nodes = nodes.split("/")
        # 1. Search & get new child.
        added, exists = None, False
        # (1) Search.
        for child in self._children:
            if child._name == nodes[0]:
                added, exists = child, True
                break
        # (2) Get child.
        if not exists:
            added = _TreeNode(name=nodes[0])
            added._parent, added._index_in_brother = self, len(self._children)
        # 2. Add residual nodes for new child.
        if len(nodes) > 1: added.add_child(nodes[1:])
        """ Add child. """
        if not exists: self._children.append(added)

    def get_child(self, nodes):
        if isinstance(nodes, str): nodes = nodes.split("/")
        # Search for current child.
        for child in self._children:
            if child._name == nodes[0]:
                if len(nodes) == 1: return child
                else: return child.get_child(nodes[1:])
        raise ValueError("Cannot find child '%s'" % str(nodes))


def get_xlwt_style(font_color='black', is_font_bold=False, border_cfgs=None, is_middle_align=True):
    """
    :param font_color: Str: 'black', 'red', 'blue', 'purple'
    :param is_font_bold: True or False.
    :param border_cfgs: Dict of { position: style }, where
        - position: 'top', 'bottom', 'left', 'right', 'all'
        - style: 'medium', 'double'.
    :param is_middle_align: True or False.
    :return:
    """
    style = xlwt.XFStyle()
    # 1. Font.
    if font_color != 'black': style.font.colour_index = {'red': 2, 'blue': 4, 'purple': 6}[font_color]
    style.font.bold = is_font_bold
    # 2. Border.
    if border_cfgs is not None:
        for pos, line_style in border_cfgs.items():
            line_style = {'medium': xlwt.Borders.MEDIUM, 'double': xlwt.Borders.DOUBLE}[line_style]
            if pos in ['top', 'all']: style.borders.top = line_style
            elif pos in ['bottom', 'all']: style.borders.bottom = line_style
            elif pos in ['left', 'all']: style.borders.left = line_style
            elif pos in ['right', 'all']: style.borders.right = line_style
            else: raise ValueError("Invalid position '%s'. " % pos)
    # 3. Middle align
    if is_middle_align:
        style.alignment.vert = 0x01
        style.alignment.horz = 0x02
    # Return
    return style


def summarize_to_xls(records, save_dir, save_name):
    """
    :param records: A list of {title1: v1, title2: v2, ...}
    :param save_dir:
    :param save_name:
    :return:
    """
    # Generate book & sheet.
    book = xlwt.Workbook()
    sheet = book.add_sheet('sheet1')
    # Build tree for titles.
    titles_root = _TreeNode()
    for rec in records:
        for k in rec.keys(): titles_root.add_child(k)
    ####################################################################################################################
    # 1. Write titles.
    ####################################################################################################################

    def _dfs_write(_root):
        for _child in _root.children:
            # 1. Get style for current child.
            border_cfgs = {'top': 'medium', 'bottom': 'medium', 'left': 'medium', 'right': 'medium'}
            if _child.row_index == 1: border_cfgs['top'] = 'double'
            if _child.row_index + _child.row_expand == titles_root.depth_subtree: border_cfgs['bottom'] = 'double'
            # 2. Write
            sheet.write_merge(
                _child.row_index, _child.row_index+_child.row_expand-1, _child.col_index, _child.col_index+_child.col_expand-1, _child.name,
                style=get_xlwt_style(is_font_bold=True, border_cfgs=border_cfgs, is_middle_align=True))
            # DFS.
            _dfs_write(_child)

    _dfs_write(_root=titles_root)
    ####################################################################################################################
    # 2. Write values.
    ####################################################################################################################
    for index, rec in enumerate(records):
        for k, v in rec.items():
            row_index = titles_root.depth_subtree + index
            col_index = titles_root.get_child(nodes=k).col_index
            """ Write """
            sheet.write(row_index, col_index, v, style=get_xlwt_style(is_middle_align=True))

    # Save
    book.save(os.path.join(save_dir, '%s.xls' % save_name))
