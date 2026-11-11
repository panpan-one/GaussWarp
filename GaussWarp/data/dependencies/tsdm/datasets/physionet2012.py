__all__ = ['Physionet2012']
import os
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from data.dependencies.tsdm.datasets.base import DATASET_OBJECT, MultiFrameDataset
from data.dependencies.tsdm.encoders import TripletDecoder
from data.dependencies.tsdm.utils.types import PathType
GENERAL_DESCRIPTORS = {'Age': int, 'Gender': int, 'Height': float, 'ICUType': int, 'Weight': float}
variables = {'pH': np.nan, 'PaCO2': np.nan, 'PaO2': np.nan, 'FiO2': np.nan, 'MechVent': np.nan, 'DiasABP': np.nan, 'HR': np.nan, 'MAP': np.nan, 'SysABP': np.nan, 'Temp': np.nan, 'GCS': np.nan, 'Urine': np.nan, 'Weight': np.nan, 'HCT': np.nan, 'BUN': np.nan, 'Creatinine': np.nan, 'Glucose': np.nan, 'HCO3': np.nan, 'Mg': np.nan, 'Platelets': np.nan, 'K': np.nan, 'Na': np.nan, 'WBC': np.nan, 'NIDiasABP': np.nan, 'NIMAP': np.nan, 'NISysABP': np.nan, 'RespRate': np.nan, 'ALP': np.nan, 'ALT': np.nan, 'AST': np.nan, 'Bilirubin': np.nan, 'SaO2': np.nan, 'Lactate': np.nan, 'Albumin': np.nan, 'TroponinT': np.nan, 'Cholesterol': np.nan, 'TroponinI': np.nan}
countn = {'pH': 0, 'PaCO2': 0, 'PaO2': 0, 'FiO2': 0, 'MechVent': 0, 'DiasABP': 0, 'HR': 0, 'MAP': 0, 'SysABP': 0, 'Temp': 0, 'GCS': 0, 'Urine': 0, 'Weight': 0, 'HCT': 0, 'BUN': 0, 'Creatinine': 0, 'Glucose': 0, 'HCO3': 0, 'Mg': 0, 'Platelets': 0, 'K': 0, 'Na': 0, 'WBC': 0, 'NIDiasABP': 0, 'NIMAP': 0, 'NISysABP': 0, 'RespRate': 0, 'ALP': 0, 'ALT': 0, 'AST': 0, 'Bilirubin': 0, 'SaO2': 0, 'Lactate': 0, 'Albumin': 0, 'TroponinT': 0, 'Cholesterol': 0, 'TroponinI': 0}
time_range = np.arange(0, 49)

def _append_record(d: dict[str, list], r: Mapping) -> dict[str, list]:
    for key in d:
        d[key].append(r[key])
    return d

def read_physionet_record(f: IO[bytes], unravel_triplets: bool=False) -> tuple[int, int, dict, DATASET_OBJECT]:
    lines = iter(f)
    _ = next(lines)
    general_descriptors = {}
    record_id = None
    ICU_type = None
    timestamps: list[int] = []
    timeseries: dict[str, Any] = variables.copy()
    counts = countn.copy()
    rows: list[dict[str, float]] = []
    row_times: list[np.int32] = []
    prev_ts = -1.0
    first = True
    for line in lines:
        time, parameter_bytes, value_bytes = (token.strip() for token in line.split(b','))
        hours, minutes = time.split(b':')
        time_minutes = int(hours) * 60 + int(minutes)
        parameter = parameter_bytes.decode(encoding='utf-8')
        if parameter == 'RecordID':
            record_id = int(value_bytes.decode())
        elif parameter in GENERAL_DESCRIPTORS and time_minutes == 0:
            v = GENERAL_DESCRIPTORS[parameter](value_bytes.decode())
            general_descriptors[parameter] = v if v > -1 else np.nan
        else:
            value = float(value_bytes.decode())
            if prev_ts == time_minutes // 60 or first == True:
                if first:
                    prev_ts = time_minutes // 60
                if parameter not in timeseries:
                    continue
                if np.isnan(timeseries[parameter]):
                    timeseries[parameter] = 0
                timeseries[parameter] += value
                counts[parameter] += 1
                first = False
            else:
                rows.append({key: timeseries[key] / counts[key] if counts[key] > 0 else np.nan for key in variables})
                row_times.append(np.int32(prev_ts))
                prev_ts = time_minutes // 60
                timestamps: list[int] = []
                timeseries: dict[str, Any] = variables.copy()
                counts = countn.copy()
    df = pd.DataFrame(rows, index=pd.Index(row_times), columns=list(variables), dtype=float)
    if record_id is None:
        raise ValueError('RecordID not found')
    for key in GENERAL_DESCRIPTORS:
        assert key in general_descriptors, 'incomplete metadata'
    df.index.name = 'Time'
    return (record_id, ICU_type, general_descriptors, df)

class Physionet2012(MultiFrameDataset):
    BASE_URL = 'https://archive.physionet.org/challenge/2012/'
    INFO_URL = 'https://archive.physionet.org/challenge/2012/'
    RAWDATA_SHA256 = {'set-a.tar.gz': '8cb250f179cd0952b4b9ebcf8954b63d70383131670fac1cfee13deaa13ca920', 'set-b.tar.gz': 'b1637a2a423a8e76f8f087896cfc5fdf28f88519e1f4e874fbda69b2a64dac30', 'set-c.tar.gz': 'a4a56b95bcee4d50a3874fe298bf2998f2ed0dd98a676579573dc10419329ee1'}
    rawdata_files = {'A': 'set-a.tar.gz', 'B': 'set-b.tar.gz', 'C': 'set-c.tar.gz'}
    rawdata_paths: dict[str, Path]
    unravel_triplets: bool

    @property
    def dataset_files(self):
        postfix = 'triplet' if self.unravel_triplets else 'sparse'
        return {'A': f'Physionet2012-set-A-{postfix}.tar', 'B': f'Physionet2012-set-B-{postfix}.tar', 'C': f'Physionet2012-set-C-{postfix}.tar'}

    def __init__(self, *, unravel_triplets: bool=False):
        self.unravel_triplets = unravel_triplets
        super().__init__()

    @property
    def index(self) -> list:
        return list(self.dataset_files.keys())

    def _clean(self, key):
        record_ids_list = []
        metadata: dict[str, list[float]] = {key: [] for key in GENERAL_DESCRIPTORS}
        time_series = []
        with tarfile.open(str(self.rawdata_paths[key]), 'r:gz') as archive, tqdm(archive.getmembers()) as progress_bar:
            for member in progress_bar:
                progress_bar.set_description(f'Loading patient data {member}')
                if not member.isreg():
                    continue
                with archive.extractfile(member) as record_f:
                    record_id, ICU_type, descriptors, observations = read_physionet_record(record_f, unravel_triplets=self.unravel_triplets)
                    record_ids_list.append(record_id)
                    for k in metadata:
                        metadata[k].append(descriptors[k])
                    time_series.append(observations)
        record_ids = np.array(record_ids_list, dtype=np.int64)
        record_ids_unraveled = np.repeat(record_ids, repeats=[len(ts) for ts in time_series])
        timestamps_unraveled = np.concatenate([ts.index for ts in time_series])
        metadata_df = pd.DataFrame(metadata, index=pd.Index(record_ids, name='RecordID'))
        time_series_df = pd.concat(time_series, ignore_index=True)
        time_series_df.set_index(pd.MultiIndex.from_arrays((record_ids_unraveled, timestamps_unraveled), names=('RecordID', 'Time')), inplace=True)
        time_series_df.columns.name = None
        self.dataset[key] = (time_series_df, metadata_df)
        return (metadata_df, time_series_df)

    @staticmethod
    def serialize(frames: tuple[DATASET_OBJECT, DATASET_OBJECT], path: PathType, /, **kwargs: Any) -> None:
        metadata, series = frames
        with tarfile.open(path, mode='w') as archive:
            with tempfile.TemporaryDirectory(prefix='tsdm') as tmp_dir:
                path_metadata = os.path.join(tmp_dir, 'metadata.feather')
                metadata.reset_index().to_feather(path_metadata)
                archive.add(name=path_metadata, arcname='/metadata.feather')
                path_series = os.path.join(tmp_dir, 'series.feather')
                series.reset_index().to_feather(path_series)
                archive.add(name=path_series, arcname='/series.feather')
        archive.close()

    @staticmethod
    def deserialize(path: PathType, /, *, squeeze: bool=True) -> tuple[pd.DataFrame, pd.DataFrame]:
        with tarfile.open(path, mode='r') as archive:
            with archive.extractfile(archive.getmember('metadata.feather')) as meta_f:
                metadata = pd.read_feather(meta_f)
            metadata.set_index(keys='RecordID', drop=True, inplace=True)
            with archive.extractfile(archive.getmember('series.feather')) as series_f:
                series = pd.read_feather(series_f)
            series.set_index(['RecordID', 'Time'], drop=True, inplace=True)
            if squeeze:
                metadata = metadata.squeeze()
                series = series.squeeze()
            return (series, metadata)
