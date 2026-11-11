import os
import torch
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class Exp_Basic(object):

    def __init__(self, configs: ExpConfigs):
        self.configs = configs
        self.device = self._acquire_device()
        self.retain_graph = False

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.configs.use_gpu:
            if self.configs.use_multi_gpu:
                if self.configs.gpu_ids is not None:
                    device_ids = self.configs.gpu_ids.replace(' ', '').split(',')
                    device_ids = [int(id_) for id_ in device_ids]
                    self.configs.gpu_id = device_ids[0]
                    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
                    logger.debug(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', '')}")
                else:
                    logger.warning(f"Trying to use accelerate's multi gpu settings. Since --gpu_ids argument is not given.")
            else:
                logger.debug(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', '')}")
            device = torch.device(f'cuda:{self.configs.gpu_id}')
            logger.debug(f'Primary GPU: {self.configs.gpu_id}')
        else:
            device = torch.device('cpu')
            logger.warning('GPU is not available. Check your Pytorch installation.')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
