
import os
from custom_pkg.io.config import CanonicalConfigPyTorch


########################################################################################################################
# Config for train.
########################################################################################################################

class ConfigTrain(CanonicalConfigPyTorch):
    """ The config for training. """
    def __init__(self, exp_dir=os.path.join(os.path.split(os.path.realpath(__file__))[0], '../STORAGE/experiments'), **kwargs):
        super(ConfigTrain, self).__init__(exp_dir, **kwargs)

    def _set_directory_args(self, **kwargs):
        super(ConfigTrain, self)._set_directory_args()
        self.args.eval_quali_dir = os.path.join(self.args.ana_train_dir, 'quali')
        self.args.eval_quant_dir = os.path.join(self.args.ana_train_dir, 'quant')

    def _add_root_args(self):
        super(ConfigTrain, self)._add_root_args()
        self.parser.add_argument("--dataset",                                   type=str,  default='shapes2d')

    def _add_tree_args(self, args_dict):
        ################################################################################################################
        # Datasets.
        ################################################################################################################
        # Dataset configs.
        if args_dict['dataset'] in ['shapes2d', 'faces', 'small_norb']:
            self.parser.set(['input_nc', 'input_size'], [1, 64])
        if args_dict['dataset'] in ['shapes3d', 'cars3d']:
            self.parser.set(['input_nc', 'input_size'], [3, 64])
        self.parser.add_argument("--dataset_drop_last",                         type=bool,  default=True)
        self.parser.add_argument("--dataset_shuffle",                           type=bool,  default=True)
        self.parser.add_argument("--dataset_num_threads",                       type=int,   default=0)
        ################################################################################################################
        # Modules
        ################################################################################################################
        self.parser.add_argument("--nz",                                        type=int,   default=10)
        self.parser.add_argument("--init_size",                                 type=int,   default=4)
        self.parser.add_argument("--ncs",                                       type=int,   nargs='+',  default=[128, 64, 32, 16, 4])
        self.parser.add_argument("--hidden_ncs",                                type=int,   nargs='+',  default=[256, 128, 64, 32, 16])
        self.parser.add_argument("--middle_ns_flows",                           type=int,   nargs='+',  default=[1, 1, 1, 1])
        ################################################################################################################
        # Optimization.
        ################################################################################################################
        self.parser.add_argument("--sigma",                                     type=float, default=1.0)
        self.parser.add_argument("--random_uni_radius",                         type=float, default=1.0)
        # Lambda
        self.parser.add_argument("--lambda_recon",                              type=float, default=1.0)
        self.parser.add_argument("--lambda_kld",                                type=float, default=1000.0)
        self.parser.add_argument("--lambda_jacob_sn",                           type=float, default=1000.0)
        self.parser.add_argument("--lambda_jacob_sv",                           type=float, default=0.01)
        self.parser.add_argument("--sn_power",                                  type=int,   default=5)
        self.parser.add_argument("--lambda_norm_capacity",                      type=float, default=10.0)
        self.parser.add_argument("--lambda_norm_sv",                            type=float, default=0.01)
        ################################################################################################################
        # Evaluation.
        ################################################################################################################
        # Quali.
        self.parser.add_argument("--freq_counts_step_eval_quali",               type=int,   default=50)
        self.parser.add_argument("--eval_quali_recon_sqrt_num",                 type=int,   default=8)
        self.parser.add_argument("--eval_quali_trav_n",                         type=int,   default=10)
        self.parser.add_argument("--eval_quali_sv_n",                           type=int,   default=10000)
        self.parser.add_argument("--eval_quali_sv_batch_size",                  type=int,   default=1024)
        # Quant.
        self.parser.add_argument("--freq_counts_step_eval_quant",               type=int,   default=2)
        self.parser.add_argument("--eval_quant_my_z_batch_size",                type=int,   default=2560)
        self.parser.add_argument("--eval_quant_my_z_max_counts",                type=int,   default=10000)
        self.parser.add_argument("--eval_quant_my_z_nbins",                     type=int,   nargs='+',  default=[3, 5, 10, 20, 50])
        # Jacob.
        self.parser.add_argument("--freq_counts_step_eval_jacob",               type=int,   default=5)
        self.parser.add_argument("--eval_jacob_n",                              type=int,   default=100)
        self.parser.add_argument("--eval_jacob_batch_size",                     type=int,   default=4)
        self.parser.add_argument("--eval_jacob_ag_bsize",                       type=int,   default=512)

    def _add_additional_args(self):
        # Epochs & batch size
        self.parser.add_argument("--steps",                         type=int,   default=100000)
        self.parser.add_argument("--batch_size",                    type=int,   default=64)
        # Learning rate
        self.parser.add_argument("--learning_rate",                 type=float, default=0.0001)
        # Frequency
        self.parser.add_argument("--freq_counts_step_log",          type=int,   default=1000)
        self.parser.add_argument("--freq_counts_step_chkpt",        type=int,   default=1)
