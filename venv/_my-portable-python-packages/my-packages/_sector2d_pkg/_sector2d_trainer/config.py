
import os
from ._custom_pkg.io.config import CanonicalConfigPyTorch


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

    def _add_tree_args(self, args_dict):
        ################################################################################################################
        # Optimization
        ################################################################################################################
        # Lambda
        self.parser.add_argument("--lambda_2sector_recon",                      type=float, default=1.0)
        self.parser.add_argument("--lambda_2sector_om",                         type=float, default=1.0)
        self.parser.add_argument("--lambda_2sector_unif",                       type=float, default=10.0)
        self.parser.add_argument("--lambda_2latent_recon",                      type=float, default=1000.0)
        self.parser.add_argument("--lambda_2latent_om",                         type=float, default=1000.0)
        self.parser.add_argument("--lambda_2latent_unif",                       type=float, default=10.0)
        ################################################################################################################
        # Evaluation
        ################################################################################################################
        # Qualitative.
        self.parser.add_argument("--freq_counts_step_eval_quali",               type=int,   default=200)
        self.parser.add_argument("--eval_quali_rec_n",                          type=int,   default=1000)
        self.parser.add_argument("--eval_quali_crsp_n_grids",                   type=int,   default=10)

    def _add_additional_args(self):
        # Epochs & batch size
        self.parser.add_argument("--steps",                         type=int,   default=100000)
        self.parser.add_argument("--batch_size",                    type=int,   default=64)
        # Learning rate
        self.parser.add_argument("--learning_rate",                 type=float, default=0.0001)
        # Frequency
        self.parser.add_argument("--freq_counts_step_log",          type=int,   default=1000)
        self.parser.add_argument("--freq_counts_step_chkpt",        type=int,   default=1)
