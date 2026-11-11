__all__ = ['MIMIC_III_DeBrouwer2019']
from pathlib import Path
import os
import pandas as pd
from data.dependencies.tsdm.datasets.base import MultiFrameDataset
from data.dependencies.tsdm.encoders import TripletDecoder

class MIMIC_III_DeBrouwer2019(MultiFrameDataset):
    BASE_URL = 'https://physionet.org/content/mimiciii/get-zip/1.4/'
    INFO_URL = 'https://physionet.org/content/mimiciii/1.4/'
    HOME_URL = 'https://mimic.mit.edu/'
    GITHUB_URL = 'https://github.com/edebrouwer/gru_ode_bayes/'
    RAWDATA_SHA256 = '8e884a916d28fd546b898b54e20055d4ad18d9a7abe262e15137080e9feb4fc2'
    RAWDATA_SHAPE = (3082224, 7)
    QINGDAO_COMPATIBLE_RAWDATA_SHAPE = (3082464, 7)
    DATASET_SHA256 = {'timeseries': '2ebb7da820560f420f71c0b6fb068a46449ef89b238e97ba81659220fae8151b', 'metadata': '4779aa3639f468126ea263645510d5395d85b73caf1c7abb0a486561b761f5b4'}
    DATASET_SHAPE = {'timeseries': (552327, 96), 'metadata': (96, 3)}
    dataset_files = {'timeseries': 'timeseries.parquet', 'metadata': 'metadata.parquet'}
    rawdata_files = 'complete_tensor.csv'
    rawdata_paths: Path
    index = ['timeseries', 'metadata']

    def _clean(self, key):
        if not self.rawdata_paths.exists():
            raise RuntimeError(f"Please apply the preprocessing code found at {self.GITHUB_URL}.\nPut the resulting file 'complete_tensor.csv' in {self.RAWDATA_DIR}.")
        ts = pd.read_csv(self.rawdata_paths, index_col=0)
        allow_qingdao_compatible = os.environ.get('HCO_MIMIC_ALLOW_QINGDAO_COMPATIBLE', '0') == '1'
        accepted_shapes = {self.RAWDATA_SHAPE}
        if allow_qingdao_compatible:
            accepted_shapes.add(self.QINGDAO_COMPATIBLE_RAWDATA_SHAPE)
        if ts.shape not in accepted_shapes:
            raise ValueError(f'The ts.shape={ts.shape!r} is not accepted. Paper-exact preprocessing requires (3082224, 7). The explicitly enabled qingdao/pandas-2 compatibility profile requires (3082464, 7).')
        ts = ts.sort_values(by=['UNIQUE_ID', 'TIME_STAMP'])
        ts = ts.astype({'UNIQUE_ID': 'int16', 'TIME_STAMP': 'int16', 'LABEL_CODE': 'int16', 'VALUENORM': 'float32', 'MEAN': 'float32', 'STD': 'float32'})
        means = ts.groupby('LABEL_CODE').mean()['VALUENUM'].rename('MEANS')
        stdvs = ts.groupby('LABEL_CODE').std()['VALUENUM'].rename('STDVS')
        stats = pd.DataFrame([means, stdvs]).T.reset_index()
        stats = stats.astype({'LABEL_CODE': 'int16', 'MEANS': 'float32', 'STDVS': 'float32'})
        ts = ts[['UNIQUE_ID', 'TIME_STAMP', 'LABEL_CODE', 'VALUENORM']]
        ts = ts.reset_index(drop=True)
        ts = ts.set_index(['UNIQUE_ID', 'TIME_STAMP'])
        ts = ts.sort_index()
        encoder = TripletDecoder(value_name='VALUENORM', var_name='LABEL_CODE')
        encoder.fit(ts)
        ts = encoder.encode(ts)
        ts.columns = ts.columns.astype('string')
        stats.to_parquet(self.dataset_paths['metadata'])
        ts.to_parquet(self.dataset_paths['timeseries'])

    def _load(self, key):
        return pd.read_parquet(self.dataset_paths[key])

    def _download(self, **kwargs):
        if not self.rawdata_paths.exists():
            raise RuntimeError(f"Please apply the preprocessing code found at {self.GITHUB_URL}.\nPut the resulting file 'complete_tensor.csv' in {self.RAWDATA_DIR}.")
