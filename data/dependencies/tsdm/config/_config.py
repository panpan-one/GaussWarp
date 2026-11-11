__all__ = ['CONFIG', 'DATASETS', 'MODELS', 'HASHES', 'HOMEDIR', 'BASEDIR', 'LOGDIR', 'MODELDIR', 'DATASETDIR', 'RAWDATADIR', 'DEFAULT_DEVICE', 'DEFAULT_DTYPE', 'conf', 'Config']
import logging
import os
from importlib import resources
from pathlib import Path
import torch
import yaml
from data.dependencies.tsdm.config import config_files
__logger__ = logging.getLogger(__name__)
os.environ['TSDM_AUTOJIT'] = 'True'
NAME = __name__
FILE = __file__

class Config:
    _autojit: bool = True
    __name__ = NAME
    __file__ = FILE

    @property
    def autojit(self) -> bool:
        return self._autojit

    @autojit.setter
    def autojit(self, value: bool) -> None:
        assert isinstance(value, bool)
        self._autojit = bool(value)
        os.environ['TSDM_AUTOJIT'] = str(value)
conf: Config = Config()
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_DTYPE = torch.float32
with resources.open_text(config_files, 'config.yaml') as file:
    CONFIG = yaml.safe_load(file)
with resources.open_text(config_files, 'models.yaml') as file:
    MODELS = yaml.safe_load(file)
with resources.open_text(config_files, 'datasets.yaml') as file:
    DATASETS = yaml.safe_load(file)
with resources.open_text(config_files, 'hashes.yaml') as file:
    HASHES = yaml.safe_load(file)
HOMEDIR = Path.home()
BASEDIR = Path(os.environ.get('GAUSSWARP_TSDM_ROOT', str(HOMEDIR.joinpath(CONFIG['basedir'])))).expanduser()
LOGDIR = BASEDIR.joinpath(CONFIG['logdir'])
MODELDIR = BASEDIR.joinpath(CONFIG['modeldir'])
DATASETDIR = BASEDIR.joinpath(CONFIG['datasetdir'])
RAWDATADIR = BASEDIR.joinpath(CONFIG['rawdatadir'])
LOGDIR.mkdir(parents=True, exist_ok=True)
__logger__.info('Available Models: %s', set(MODELS))
__logger__.info('Available Datasets: %s', set(DATASETS))

def generate_folders(d: dict, current_path: Path) -> None:
    for directory in d:
        path = current_path.joinpath(directory)
        if d[directory] is None:
            __logger__.debug('creating folder %s', path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            generate_folders(d[directory], path)
__logger__.debug('Initializing folder structure')
generate_folders(CONFIG['folders'], BASEDIR)
__logger__.debug('Created folder structure in %s', BASEDIR)
