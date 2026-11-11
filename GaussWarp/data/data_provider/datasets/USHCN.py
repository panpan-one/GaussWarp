import warnings
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, ConcatDataset
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs
from utils.configs import configs
from data.dependencies.tsdm.tasks.ushcn_debrouwer2019 import USHCN_DeBrouwer2019
from data.dependencies.tsdm.tasks.ushcn_debrouwer2019 import Sample
warnings.filterwarnings('ignore')

class Data(Dataset):

    def __init__(self, configs: ExpConfigs, flag: str='train', **kwargs):
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
        if self.seq_len + self.pred_len > 200:
            raise ValueError('The requested history and prediction lengths exceed the supported window.')
        task = USHCN_DeBrouwer2019(seq_len=self.seq_len - 0.5, pred_len=self.pred_len)
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
    L_TOTAL = 200
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
    seq_len_max_irr: int = configs.seq_len_max_irr
    pred_len_max_irr: int = configs.pred_len_max_irr
    xs: list[Tensor] = []
    ys: list[Tensor] = []
    x_marks: list[Tensor] = []
    y_marks: list[Tensor] = []
    x_masks: list[Tensor] = []
    y_masks: list[Tensor] = []
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
        sample_IDs.append(sample_ID)
    ENC_IN = xs[0].shape[-1]
    xs.append(torch.zeros(seq_len_max_irr, ENC_IN))
    x_marks.append(torch.zeros(seq_len_max_irr))
    x_masks.append(torch.zeros(seq_len_max_irr, ENC_IN))
    ys.append(torch.zeros(pred_len_max_irr, ENC_IN))
    y_marks.append(torch.zeros(pred_len_max_irr))
    y_masks.append(torch.zeros(pred_len_max_irr, ENC_IN))
    xs = pad_sequence(xs, batch_first=True, padding_value=float('nan'))
    x_marks = pad_sequence(x_marks, batch_first=True, padding_value=float('nan'))
    x_masks = pad_sequence(x_masks, batch_first=True)
    ys = pad_sequence(ys, batch_first=True, padding_value=float('nan'))
    y_marks = pad_sequence(y_marks, batch_first=True, padding_value=float('nan'))
    y_masks = pad_sequence(y_masks, batch_first=True)
    xs = xs[:-1]
    x_marks = x_marks[:-1]
    x_masks = x_masks[:-1]
    ys = ys[:-1]
    y_marks = y_marks[:-1]
    y_masks = y_masks[:-1]
    sample_IDs = torch.tensor(sample_IDs).float()
    return {'x': torch.nan_to_num(xs), 'x_mark': fix_nan_x_mark(x_marks.unsqueeze(-1), seq_len=configs.seq_len).float(), 'x_mask': x_masks.float(), 'y': torch.nan_to_num(ys), 'y_mark': fix_nan_y_mark(y_marks.unsqueeze(-1)).float(), 'y_mask': y_masks.float(), 'sample_ID': sample_IDs}
