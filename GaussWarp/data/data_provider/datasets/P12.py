import warnings
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, ConcatDataset
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs
from utils.configs import configs
from data.dependencies.tsdm.tasks.P12 import Physionet2012
from data.dependencies.tsdm.tasks.P12 import Sample
warnings.filterwarnings('ignore')

class Data(Dataset):

    def __init__(self, configs: ExpConfigs, flag: str='train', **kwargs):
        logger.debug(f"getting {flag} set of PhysioNet'12 in tsdm format")
        self.configs = configs
        assert flag in ['train', 'test', 'val', 'test_all']
        self.flag = flag
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.preprocess()

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def preprocess(self):
        if self.seq_len + self.pred_len > 96:
            raise ValueError('The requested history and prediction lengths exceed the supported window.')
        task = Physionet2012(seq_len=self.seq_len - 0.5, pred_len=self.pred_len)
        if self.flag != 'val':
            dataset = ConcatDataset([task.get_dataset((0, 'train')), task.get_dataset((0, 'val'))])
            self.seq_len_max_irr = 0
            self.pred_len_max_irr = 0
            for sample in dataset:
                x_mark, x, y_mark = sample.inputs
                y = sample.targets
                self.seq_len_max_irr = max(self.seq_len_max_irr, x.shape[0])
                self.pred_len_max_irr = max(self.pred_len_max_irr, y.shape[0])
            self.configs.seq_len_max_irr = self.seq_len_max_irr
            self.configs.pred_len_max_irr = self.pred_len_max_irr
        if self.flag == 'test_all':
            dataset = ConcatDataset([task.get_dataset((0, 'train')), task.get_dataset((0, 'val')), task.get_dataset((0, 'test'))])
        else:
            dataset = task.get_dataset((0, self.flag))
        self.dataset = dataset

def fix_nan_x_mark(x_mark, seq_len):
    L_TOTAL = 48
    BATCH_SIZE, SEQ_LEN_MAX_IRR, _ = x_mark.shape
    indices = torch.linspace(start=seq_len / L_TOTAL - 2 * 0.01, end=seq_len / L_TOTAL - 0.001, steps=SEQ_LEN_MAX_IRR).to(x_mark.device).view(1, -1, 1).repeat(BATCH_SIZE, 1, 1)
    nan_mask = torch.isnan(x_mark)
    x_mark[nan_mask] = indices[nan_mask]
    return x_mark

def fix_nan_y_mark(y_mark):
    BATCH_SIZE, PRED_LEN, _ = y_mark.shape
    indices = torch.linspace(start=1 - 2 * 0.01, end=1 - 0.001, steps=PRED_LEN).to(y_mark.device).view(1, -1, 1).repeat(BATCH_SIZE, 1, 1)
    nan_mask = torch.isnan(y_mark)
    y_mark[nan_mask] = indices[nan_mask]
    return y_mark

def collate_fn(batch: list[Sample]) -> dict[Tensor]:
    global configs
    xs: list[Tensor] = []
    ys: list[Tensor] = []
    x_marks: list[Tensor] = []
    y_marks: list[Tensor] = []
    x_masks: list[Tensor] = []
    y_masks: list[Tensor] = []
    query_masks: list[Tensor] = []
    sample_IDs: list[int] = []
    for sample in batch:
        x_mark, x, y_mark = sample.inputs
        y = sample.targets
        sample_ID = sample.key
        y_mask = y.isfinite()
        x_mask = x.isfinite()
        xs.append(x)
        x_marks.append(x_mark)
        x_masks.append(x_mask)
        ys.append(y)
        y_marks.append(y_mark)
        y_masks.append(y_mask)
        query_masks.append(torch.ones(len(y_mark), dtype=torch.bool, device=y_mark.device))
        sample_IDs.append(sample_ID)
    ENC_IN = xs[0].shape[-1]
    xs = pad_sequence(xs, batch_first=True, padding_value=float('nan'))
    x_marks = pad_sequence(x_marks, batch_first=True)
    x_masks = pad_sequence(x_masks, batch_first=True)
    ys = pad_sequence(ys, batch_first=True, padding_value=float('nan'))
    cutoff = (configs.seq_len - 0.5) / 48.0
    y_marks = pad_sequence(y_marks, batch_first=True, padding_value=cutoff)
    y_masks = pad_sequence(y_masks, batch_first=True)
    query_masks = pad_sequence(query_masks, batch_first=True)
    if ys.shape[1] == 0:
        ys = xs.new_full((len(batch), 1, ENC_IN), float('nan'))
        y_masks = xs.new_zeros((len(batch), 1, ENC_IN))
        y_marks = xs.new_full((len(batch), 1), cutoff)
        query_masks = torch.zeros(len(batch), 1, dtype=torch.bool, device=xs.device)
    sample_IDs = torch.tensor(sample_IDs).float()
    return {'x': torch.nan_to_num(xs), 'x_mark': x_marks.unsqueeze(-1).float(), 'x_mask': x_masks.float(), 'y': torch.nan_to_num(ys), 'y_mark': y_marks.unsqueeze(-1).float(), 'y_mask': y_masks.float(), 'query_mask': query_masks.unsqueeze(-1), 'sample_ID': sample_IDs}
