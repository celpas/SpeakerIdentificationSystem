import os
import numpy as np
import logging
import config as cfg

from models.snr_estimation.default_args import *
from models.snr_estimation.model import DeepXi
from models.snr_estimation.prelim import Prelim
import models.snr_estimation.utils as utils
from models.snr_estimation.se_batch import BatchV2

logger = logging.getLogger(__name__)


class SNRManager:
    
    def __init__(self, device='cpu'):
        self.device = device
        self.weights = cfg.WEIGHTS_DEEPXI
        if not os.path.exists(self.weights):
            logger.error('Weights not found!')
        else:
            self.load_model()
            
    def load_model(self):
        self._load_args()
        self.args.gpu = 1 if self.device == 'cuda' else 0
        self.args.padding = 'causal' if self.args.causal else 'same'
        self.args.model_path = self.weights
        
        config = utils.gpu_config(self.args.gpu)
        
        N_d = int(self.args.f_s*self.args.T_d*0.001)        # window duration (samples)
        N_s = int(self.args.f_s*self.args.T_s*0.001)        # window shift (samples)
        K   = int(pow(2, np.ceil(np.log2(N_d))))  # number of DFT components
        
        self.model = DeepXi(
            N_d=N_d,
            N_s=N_s,
            K=K,
            #sample_dir=args.data_path,
            **vars(self.args)
        )
        logger.debug('DeepXi loaded. Version: %s', self.args.ver)
        
    def compute_snr(self, filepaths=[]):
        if len(filepaths) == 0:
            logger.error('No files to process!')
            return
        
        test_x, test_x_len, _, test_x_base_names = BatchV2(filepaths)
        data = self.model.inferV2(
            test_x=test_x,
            test_x_len=test_x_len,
            test_epoch=self.args.test_epoch,
            model_path=self.args.model_path,
            out_type=self.args.out_type,
            gain=self.args.gain,
            n_filters=self.args.n_filters,
        )
        
        snrs = []
        for x in data:
            snr = 10*np.log10(x+1e-12)
            snr = np.mean(snr, axis=1)
            snr = np.mean(snr)
            snrs.append(snr)
        
        return np.array(snrs)
 
    def _load_args(self):
        args = "--ver               resnet-1.1n                 \
                --network           ResNetV2                    \
                --d_model           256                         \
                --n_blocks          40                          \
                --d_f               64                          \
                --k                 3                           \
                --max_d_rate        16                          \
                --causal            0                           \
                --unit_type         ReLU->LN->W+b               \
                --loss_fnc          BinaryCrossentropy          \
                --max_epochs        200                         \
                --resume_epoch      0                           \
                --test_epoch        180                         \
                --mbatch_size       8                           \
                --inp_tgt_type      MagXi                       \
                --map_type          DBNormalCDF                 \
                --sample_size       1000                        \
                --f_s               16000                       \
                --T_d               32                          \
                --T_s               16                          \
                --min_snr           -10                         \
                --max_snr           20                          \
                --snr_inter         1                           \
                --out_type          xi_hat                      \
                --save_model        1                           \
                --log_iter          0                           \
                --eval_example      1                           \
                --gain              mmse-stsa                   \
                --infer             1                           \
                --gpu               0                           \
                --data_path         data                        \
                --test_x_path       noisy                       \
                --out_path          results                     "
        self.args = get_parser().parse_args(args.split())