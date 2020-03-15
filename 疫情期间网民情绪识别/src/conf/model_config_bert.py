# coding=utf-8
# author=yphacker

import os
from conf import config

# pretrain_model_name = 'chinese_wwm_ext'
# pretrain_model_name = 'chinese_roberta_wwm_large_ext'
pretrain_model_name = 'chinese_roberta_wwm_ext'
pretrain_model_path = os.path.join(config.pretrain_model_path, pretrain_model_name)

learning_rate = 1e-5
patience_epoch = 1
adjust_lr_num = 1