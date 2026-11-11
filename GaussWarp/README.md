# GaussWarp

**GaussWarp: Time-Warped Gaussian Slot Representation Learning for Irregular Multivariate Time Series Forecasting**

This repository provides the GaussWarp implementation and the code required to train and evaluate the model on four irregular multivariate time series (IMTS) datasets.

GaussWarp consists of three main components:

- **Time-Warped Gaussian Slot Representation Learning**: combines physical timestamps and observation order to construct warped temporal coordinates, then aggregates irregular observations into Gaussian slot representations.
- **Historical Context-Enhanced Query Interaction**: incorporates historical context into variable-specific prediction queries and models inter-variable dependencies.
- **Query-to-Slot Attention for Forecasting**: retrieves relevant historical information from Gaussian slots for each prediction query and generates forecasts at the target timestamps.

The repository contains GaussWarp and its experiment pipeline, including the data loaders required by the four benchmark datasets.

## Quickstart

### 1. Environment

Python 3.10 or later is required. Install a PyTorch build compatible with your CUDA environment, then install the project dependencies.

```bash
conda create -n gausswarp python=3.11 -y
conda activate gausswarp
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Preparation

The raw datasets are not distributed with this repository. By default, the data loader stores dataset files and processed caches under `storage/tsdm/`. You can change the data and output directories with:

```bash
export GAUSSWARP_TSDM_ROOT="$PWD/storage/tsdm"
export GAUSSWARP_RESULTS_ROOT="$PWD/storage/results"
```

The expected directory layout is:

```text
storage/tsdm/
└── rawdata/
    ├── Physionet2012/
    ├── MIMIC_III_DeBrouwer2019/
    │   └── complete_tensor.csv
    ├── USHCN_DeBrouwer2019/
    └── HumanActivity/
```

**PhysioNet 2012, USHCN, and Human Activity.** The required data are downloaded and processed automatically when the corresponding dataset is loaded for the first time. An internet connection is required.

- PhysioNet 2012: https://physionet.org/content/challenge-2012/
- Human Activity: https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity

**MIMIC-III.** Access to MIMIC-III must be obtained through PhysioNet. This dataset requires manual preprocessing. Follow the preprocessing workflow provided by [GRU-ODE-Bayes](https://github.com/edebrouwer/gru_ode_bayes) to generate `complete_tensor.csv`, and place it at:

```text
$GAUSSWARP_TSDM_ROOT/rawdata/MIMIC_III_DeBrouwer2019/complete_tensor.csv
```

The raw MIMIC-III data and preprocessing scripts are not redistributed in this repository.

Dataset page: https://physionet.org/content/mimiciii/1.4/

### 3. Training and Evaluation

Run the following command from the repository root:

```bash
bash scripts/run_gausswarp.sh <DATASET> [SEED] [GPU]
```

`DATASET` can be `P12`, `MIMIC_III`, `USHCN`, or `HumanActivity`. The default random seed is `2024`, and the default GPU index is `0`.

Examples:

```bash
bash scripts/run_gausswarp.sh P12 2024 0
bash scripts/run_gausswarp.sh MIMIC_III 2024 0
bash scripts/run_gausswarp.sh USHCN 2024 0
bash scripts/run_gausswarp.sh HumanActivity 2024 0
```

Each command trains GaussWarp, selects the checkpoint according to validation performance, and evaluates the selected checkpoint on the test split. Run the command separately for each random seed when reproducing the multi-seed results.

## Experimental Settings

The provided GaussWarp runner uses the following dataset-specific settings:

| Dataset | Historical Length | Prediction Length | Variables | Batch Size | Dropout |
|---|---:|---:|---:|---:|---:|
| P12 | 36 | 3 | 36 | 16 | 0.05 |
| MIMIC-III | 72 | 3 | 96 | 8 | 0.10 |
| USHCN | 150 | 3 | 5 | 16 | 0.05 |
| Human Activity | 3000 | 300 | 12 | 32 | 0.05 |

For a fair comparison, GaussWarp and the baseline methods are evaluated using the same training, validation, and test splits. All methods use the same dataset-specific historical lengths, prediction lengths, and input-variable settings listed above. The datasets are split into training, validation, and test sets with a ratio of 80%, 10%, and 10%, respectively. Batch size, dropout, and other model-specific hyperparameters for baseline methods follow the settings in their original papers where applicable, with necessary adaptations for the forecasting task.

The provided GaussWarp runner uses AdamW with an initial learning rate of `2e-4`, weight decay of `1e-4`, and a warmup-cosine learning-rate schedule with 5 warmup epochs. The maximum number of training epochs is 200. Early stopping is based on validation performance, with a patience of 20 epochs and a minimum training duration of 40 epochs.

The default model configuration uses 12 Gaussian slots, a hidden dimension of 64, 4 attention heads, 1 slot-attention layer, and 6 lag scales. The auxiliary loss weight is set to `0.5`.

## Benchmark Results

The following table reproduces the forecasting results reported in Table III of the paper, including all 18 baseline methods and GaussWarp on four benchmark datasets. Values are reported as mean ± standard deviation over five random seeds; lower MSE and MAE indicate better performance.

| Model | USHCN MSE ↓ | USHCN MAE ↓ | Human Activity MSE ↓ | Human Activity MAE ↓ | PhysioNet MSE ↓ | PhysioNet MAE ↓ | MIMIC-III MSE ↓ | MIMIC-III MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| t-PatchGNN (ICML24) | 0.2199±0.0594 | 0.2858±0.0269 | 0.0438±0.0008 | 0.1232±0.0050 | 0.3154±0.0098 | 0.3732±0.0095 | 0.4959±0.0148 | 0.4287±0.0083 |
| Memformer (PVLDB24) | 0.1902±0.0140 | 0.2725±0.0149 | 0.0556±0.0013 | 0.1393±0.0018 | 0.3199±0.0008 | 0.3628±0.0007 | 0.5326±0.0007 | 0.4283±0.0004 |
| LiPFormer (ICDE25) | 0.1889±0.0230 | 0.2625±0.0153 | 0.0525±0.0040 | 0.1289±0.0019 | 0.3183±0.0046 | 0.3603±0.0023 | 0.5175±0.0048 | 0.4206±0.0012 |
| Hi-Patch (ICML25) | 0.2027±0.0218 | 0.2744±0.0128 | 0.0436±0.0007 | 0.1305±0.0009 | 0.3287±0.0148 | 0.3862±0.0145 | 0.5220±0.0067 | 0.4411±0.0055 |
| FOCUS (ICDE25) | 0.3535±0.3122 | 0.2967±0.0368 | 0.0542±0.0019 | 0.1354±0.0028 | 0.3153±0.0008 | 0.3675±0.0020 | 0.5140±0.0068 | 0.4350±0.0039 |
| APN (AAAI26) | 0.1599±0.0196 | 0.2608±0.0164 | 0.0426±0.0001 | 0.1256±0.0002 | 0.3076±0.0009 | 0.3622±0.0021 | 0.4375±0.0058 | 0.4001±0.0075 |
| SeFT (ICML20) | 0.1758±0.0218 | 0.2699±0.0172 | 0.0565±0.0025 | 0.1568±0.0048 | 0.3462±0.0073 | 0.4026±0.0029 | 0.5532±0.0113 | 0.4737±0.0058 |
| mTAN (ICLR21) | 0.7328±0.2663 | 0.4657±0.0654 | 0.0627±0.0013 | 0.1624±0.0042 | 0.4437±0.0098 | 0.4658±0.0045 | 0.7722±0.0286 | 0.5643±0.0115 |
| PrimeNet (AAAI23) | 0.2656±0.0079 | 0.3660±0.0145 | 3.0259±0.1266 | 1.4071±0.0355 | 0.4549±0.0105 | 0.4799±0.0077 | 0.7110±0.0066 | 0.5483±0.0060 |
| Warpformer (KDD23) | 0.1752±0.0213 | 0.2778±0.0128 | 0.1820±0.0563 | 0.3194±0.0413 | 0.3146±0.0033 | 0.3796±0.0027 | 0.6753±0.0178 | 0.5288±0.0061 |
| GRU-D (Sci. Rep. 18) | 0.1587±0.0125 | 0.2765±0.0247 | 0.1429±0.0429 | 0.2862±0.0038 | 0.3551±0.0054 | 0.4203±0.0048 | 0.5198±0.0051 | 0.4578±0.0069 |
| NeuralFlows (NeurIPS21) | 0.2441±0.0211 | 0.3527±0.0139 | 0.0722±0.0090 | 0.1281±0.0015 | 0.3568±0.0172 | 0.3971±0.0132 | 0.4890±0.0039 | 0.4195±0.0026 |
| CRU (ICML22) | 0.3650±0.0501 | 0.4106±0.0302 | 0.1085±0.0132 | 0.2133±0.0196 | 0.4855±0.0042 | 0.4657±0.0026 | 0.7731±0.0053 | 0.5412±0.0016 |
| RFN (TKDE25) | 0.2257±0.0201 | 0.3226±0.0128 | 0.0449±0.0272 | 0.1257±0.0012 | 0.3587±0.0048 | 0.4008±0.0032 | 0.7755±0.0294 | 0.5551±0.0020 |
| Raindrop (ICLR22) | 0.2076±0.0256 | 0.3275±0.0329 | 0.4611±0.1350 | 0.4929±0.0678 | 0.3629±0.0031 | 0.4192±0.0030 | 0.5370±0.0064 | 0.4628±0.0040 |
| GraFITi (AAAI24) | 0.1585±0.0131 | 0.2587±0.0113 | 0.0426±0.0005 | 0.1220±0.0011 | 0.3110±0.0076 | 0.3577±0.0088 | 0.4600±0.0037 | 0.4112±0.0042 |
| GNeuralFlow (NeurIPS24) | 0.2906±0.0242 | 0.3791±0.0113 | 0.0928±0.0135 | 0.1233±0.0003 | 0.3521±0.0020 | 0.3936±0.0016 | 0.5054±0.0120 | 0.4257±0.0072 |
| ASTGI (ICLR26) | 0.1601±0.0150 | 0.2596±0.0071 | 0.0431±0.0028 | 0.1222±0.0081 | 0.2984±0.0015 | 0.3582±0.0022 | 0.4299±0.0053 | 0.3883±0.0027 |
| **GaussWarp (Ours)** | **0.1420±0.0057** | **0.2428±0.0059** | **0.0408±0.0007** | **0.1114±0.0006** | **0.2882±0.0015** | **0.3465±0.0017** | **0.4172±0.0041** | **0.3766±0.0024** |
| **Improvement** | **10.41%** | **6.15%** | **4.23%** | **8.69%** | **3.42%** | **3.13%** | **2.95%** | **3.01%** |

The **Improvement** row reports the relative reduction achieved by GaussWarp compared with the best-performing baseline for each dataset and metric, as reported in the paper. MSE and MAE are evaluated over valid target observations using observation masks.
## Outputs

Model checkpoints and experiment outputs are saved under `storage/results/` by default. Set `GAUSSWARP_RESULTS_ROOT` to use a different output directory. Raw datasets and processed caches are stored under `GAUSSWARP_TSDM_ROOT`.

The following environment variables can be used to configure the runner:

| Variable | Default | Description |
|---|---|---|
| `GAUSSWARP_TSDM_ROOT` | `storage/tsdm` | Root directory for raw datasets and processed caches |
| `GAUSSWARP_RESULTS_ROOT` | `storage/results` | Directory for checkpoints and experiment outputs |
| `GAUSSWARP_TRAIN_EPOCHS` | `200` | Maximum number of training epochs |
| `GAUSSWARP_MIN_TRAIN_EPOCHS` | `40` | Minimum number of training epochs |
| `GAUSSWARP_PATIENCE` | `20` | Early-stopping patience |
| `GAUSSWARP_NUM_WORKERS` | `4` | Number of data-loader workers |
