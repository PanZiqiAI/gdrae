
import os
from custom_pkg.io.config import CanonicalConfigPyTorch


########################################################################################################################
# Config for Train
########################################################################################################################

class ConfigTrainModel(CanonicalConfigPyTorch):
    """
    The config for training models.
    """
    def __init__(self, exp_dir=os.path.join(os.path.split(os.path.realpath(__file__))[0], '../STORAGE/experiments'), **kwargs):
        super(ConfigTrainModel, self).__init__(exp_dir, **kwargs)

    def _set_directory_args(self):
        super(ConfigTrainModel, self)._set_directory_args()
        self.args.eval_quali_dir = os.path.join(self.args.ana_train_dir, 'quali_results')

    def _add_root_args(self):
        super(ConfigTrainModel, self)._add_root_args()
        # Dataset
        self.parser.add_argument("--dataset",                       type=str,   default='sector2d')

    def _add_tree_args(self, args_dict):
        ################################################################################################################
        # Datasets
        ################################################################################################################
        if args_dict['dataset'] in ['sector2d', 'sector2dunif']:
            self.parser.add_argument("--n_samples",                 type=int,   default=10000)
        self.parser.add_argument("--dataset_num_threads",           type=int,   default=0)
        self.parser.add_argument("--dataset_drop_last",             type=bool,  default=True)
        self.parser.add_argument("--dataset_shuffle",               type=bool,  default=True)
        ################################################################################################################
        # Modules
        ################################################################################################################
        self.parser.add_argument("--input_nc",                      type=int,   default=1024)
        self.parser.add_argument("--hidden_nc",                     type=int,   default=1024)
        self.parser.add_argument("--n_flows",                       type=int,   default=3)
        ################################################################################################################
        # Optimization
        ################################################################################################################
        self.parser.add_argument("--random_uni_radius",             type=float, default=1.0)
        # Lambda
        self.parser.add_argument("--lambda_recon",                  type=float, default=1.0)
        self.parser.add_argument("--lambda_kld",                    type=float, default=1.0)
        self.parser.add_argument("--lambda_jacob_sn",               type=float, default=1.0)
        self.parser.add_argument("--lambda_jacob_sv",               type=float, default=1.0)
        self.parser.add_argument("--sn_power",                      type=int,   default=5)
        self.parser.add_argument("--lambda_capacity",               type=float, default=1.0)
        ################################################################################################################
        # Evaluation
        ################################################################################################################
        # Qualitative.
        self.parser.add_argument("--freq_counts_step_eval_quali",   type=int,   default=100)
        self.parser.add_argument("--eval_quali_crsp_n_grids",       type=int,   default=10)
        self.parser.add_argument("--eval_quali_mu_n",               type=int,   default=10000)
        # Quantitative.
        self.parser.add_argument("--freq_counts_step_eval_quant",   type=int,   default=10)
        self.parser.add_argument("--eval_quant_n",                  type=int,   default=10000)
        self.parser.add_argument("--eval_quant_jacob_size",         type=int,   default=32)

    def _add_additional_args(self):
        # Epochs & batch size
        self.parser.add_argument("--steps",                         type=int,   default=100000)
        self.parser.add_argument("--batch_size",                    type=int,   default=64)
        # Learning rate
        self.parser.add_argument("--learning_rate",                 type=float, default=0.0001)
        # Frequency
        self.parser.add_argument("--freq_counts_step_log",          type=int,   default=1000)
        self.parser.add_argument("--freq_counts_step_chkpt",        type=int,   default=1)
