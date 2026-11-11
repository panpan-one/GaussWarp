from pathlib import Path
import datetime
import math
import warnings
import yaml
import json
from collections import OrderedDict
from dataclasses import asdict
import importlib
import numpy as np
from tqdm import tqdm
import torch
from torch import optim, Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, LRScheduler
from accelerate import load_checkpoint_in_model
from data.data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, test_params_and_flops, test_peak_gpu_memory, test_training_step_time, test_inference_step_time
from utils.metrics import metric
from utils.globals import logger, accelerator
from utils.ExpConfigs import ExpConfigs
warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):

    def __init__(self, configs: ExpConfigs):
        super(Exp_Main, self).__init__(configs)
        self._validate_training_policy()

    def _validate_training_policy(self) -> None:
        if self.configs.train_epochs < 1:
            raise ValueError('train_epochs must be positive')
        if not 0 <= self.configs.min_train_epochs <= self.configs.train_epochs:
            raise ValueError('min_train_epochs must be in [0, train_epochs]')
        if self.configs.patience < 1:
            raise ValueError('patience must be positive')
        if self.configs.learning_rate <= 0.0:
            raise ValueError('learning_rate must be positive')
        if self.configs.weight_decay < 0.0:
            raise ValueError('weight_decay must be non-negative')
        if self.configs.gradient_clip_norm < 0.0:
            raise ValueError('gradient_clip_norm must be non-negative')
        if not 0 <= self.configs.lr_warmup_epochs < self.configs.train_epochs:
            raise ValueError('lr_warmup_epochs must be in [0, train_epochs)')
        if self.configs.val_interval != 1:
            logger.warning('val_interval is forced to 1 by the complete-validation-history policy')
            self.configs.val_interval = 1

    def _build_model(self) -> Module:
        model_module = importlib.import_module('models.' + self.configs.model_name)
        model = model_module.Model(self.configs)
        return model

    def _get_data(self, flag: str) -> tuple[Dataset, DataLoader]:
        data_set, data_loader = data_provider(self.configs, flag)
        return (data_set, data_loader)

    def _select_optimizer(self, model: Module) -> optim.Optimizer:
        optimizer_class = {'Adam': optim.Adam, 'AdamW': optim.AdamW}.get(self.configs.optimizer)
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer '{self.configs.optimizer}'")
        return optimizer_class(model.parameters(), lr=self.configs.learning_rate, weight_decay=self.configs.weight_decay)

    def _select_lr_scheduler(self, optimizer: optim.Optimizer) -> LRScheduler:
        if self.configs.lr_scheduler == 'WarmupCosineLR':
            warmup_epochs = self.configs.lr_warmup_epochs
            total_epochs = self.configs.train_epochs

            def warmup_cosine(epoch: int) -> float:
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return float(epoch + 1) / float(warmup_epochs)
                if warmup_epochs == 0:
                    progress = float(epoch) / float(max(total_epochs - 1, 1))
                else:
                    progress = float(epoch - warmup_epochs + 1) / float(max(total_epochs - warmup_epochs, 1))
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine)
        elif self.configs.lr_scheduler == 'ExponentialDecayLR':
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** epoch)
        elif self.configs.lr_scheduler == 'DelayedStepDecayLR':
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 if epoch < 2 else 0.8 ** (epoch - 2))
        elif self.configs.lr_scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.configs.train_epochs, eta_min=0.0)
        elif self.configs.lr_scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.75 * self.configs.train_epochs, 0.9 * self.configs.train_epochs], gamma=0.1)
        elif self.configs.lr_scheduler == 'constant':
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        else:
            logger.exception(f"Unknown lr scheduler '{self.configs.lr_scheduler}'", stack_info=True)
            exit(1)
        return scheduler

    def _select_criterion(self) -> Module:
        loss_module = importlib.import_module('loss_fns.' + self.configs.loss)
        criterion = loss_module.Loss(self.configs)
        return criterion

    def _get_state_dict(self, path: Path) -> OrderedDict:
        logger.info(f'Loading model checkpoint from {path}')
        state_dict = torch.load(path)
        new_state_dict = OrderedDict()
        if_fixed = False
        for key, value in state_dict.items():
            if 's4' in key and (('B' in key or 'P' in key or 'w' in key) and 'weight' not in key):
                if_fixed = True
                continue
            new_state_dict[key] = value.contiguous()
        if if_fixed:
            logger.warning('Automatically fixed model state dict errors. It may cause unexpected behavior!')
        return new_state_dict

    def _check_model_outputs(self, batch: dict, outputs: dict) -> None:
        if type(outputs) is not dict:
            logger.exception(f"Expect model's forward function to return dict. Current output's data type is {type(outputs)}.", stack_info=True)
            exit(1)
        if self.configs.task_name in ['short_term_forecast', 'long_term_forecast']:
            if 'true' in outputs.keys() and (not torch.equal(batch['y'], outputs['true'])):
                logger.warning(f"Model's outputs['true'] is not equal to input's batch['y']. Please confirm that you are not using input's batch['y'] as ground truth. This is expected in some models such as diffusion.")

    def _merge_gathered_dicts(self, dicts: list[dict]) -> dict:
        merged_dict = {}
        keys_not_returned = []
        for d in dicts:
            for key, tensor in d.items():
                if type(tensor).__name__ != 'Tensor':
                    if key not in keys_not_returned:
                        keys_not_returned.append(key)
                        logger.warning(f"key={key!r} will not be gathered for metric calculation in test, since its value has data type '{type(tensor).__name__}', which is not 'Tensor'")
                    continue
                if key in merged_dict:
                    merged_dict[key] = torch.cat((merged_dict[key], tensor.detach().cpu()), dim=0)
                else:
                    merged_dict[key] = tensor.detach().cpu()
        return merged_dict

    @staticmethod
    def _concatenate_test_batches(batches: list[np.ndarray]) -> np.ndarray:
        first = batches[0]
        if any((batch.ndim != first.ndim or batch.shape[2:] != first.shape[2:] for batch in batches)):
            raise ValueError('Evaluation batches have incompatible feature dimensions')
        if first.ndim >= 2:
            steps = max((batch.shape[1] for batch in batches))
            batches = [np.pad(batch, [(0, 0), (0, steps - batch.shape[1])] + [(0, 0)] * (batch.ndim - 2), constant_values=0) if batch.shape[1] != steps else batch for batch in batches]
        return np.concatenate(batches, axis=0)

    def vali(self, model_train: Module, vali_loader: DataLoader, criterion: Module, current_epoch: int, train_stage: int) -> np.ndarray:
        weighted_loss_sum = 0.0
        evaluation_weight_sum = 0.0
        model_train.eval()
        with torch.no_grad():
            with tqdm(total=len(vali_loader), leave=False, desc='Validating') as it:
                batch: dict[str, Tensor]
                for i, batch in enumerate(vali_loader):
                    if batch[next(iter(batch))].shape[0] != self.configs.batch_size and current_epoch == 0:
                        logger.warning(f'Batch No.{i} of total {len(vali_loader)} has actual batch_size={batch[next(iter(batch))].shape[0]}, which is not the same as --batch_size={self.configs.batch_size}')
                    if 'y_mask' in batch.keys():
                        if torch.sum(batch['y_mask']).item() == 0:
                            if current_epoch == 0:
                                logger.warning(f'Batch No.{i} of total {len(vali_loader)} has no evaluation point (inferred from y_mask), thus skipping')
                            continue
                    if not self.configs.use_multi_gpu:
                        batch = {k: v.to(f'cuda:{self.configs.gpu_id}') for k, v in batch.items()}
                    outputs: dict[str, Tensor] = model_train(exp_stage='val', train_stage=train_stage, **batch)
                    loss: Tensor = criterion(exp_stage='val', model=model_train, **outputs)['loss']
                    output_mask = outputs.get('mask')
                    if output_mask is not None:
                        evaluation_weight = float(output_mask.sum().item())
                    elif 'y_mask' in batch:
                        evaluation_weight = float(batch['y_mask'].sum().item())
                    elif 'true' in outputs:
                        evaluation_weight = float(outputs['true'].numel())
                    else:
                        evaluation_weight = float(batch[next(iter(batch))].shape[0])
                    if self.configs.dataset_name == 'P12' and output_mask is not None:
                        residual = (outputs['pred'] - outputs['true']).double()
                        residual = torch.where(output_mask.bool(), residual, 0.0)
                        weighted_loss_sum += residual.square().sum().item()
                    else:
                        weighted_loss_sum += loss.item() * evaluation_weight
                    evaluation_weight_sum += evaluation_weight
                    if accelerator.is_main_process:
                        it.update()
                        it.set_postfix(loss=f'{loss.item():.2e}')
        total_loss = weighted_loss_sum / max(evaluation_weight_sum, 1.0)
        model_train.train()
        return total_loss

    def _record_validation(self, path: Path, train_stage: int, epoch: int, train_loss: float | None, validation_loss: float, learning_rate: float, early_stopping: EarlyStopping) -> None:
        record = {'train_stage': train_stage, 'epoch': epoch, 'train_loss': train_loss, 'validation_loss': float(validation_loss), 'learning_rate': float(learning_rate), 'best_validation_loss': float(early_stopping.val_loss_min), 'improved': bool(early_stopping.last_improved), 'early_stopping_counter': int(early_stopping.counter)}
        if accelerator.is_main_process:
            with open(path / 'validation_history.jsonl', 'a', encoding='utf-8') as history:
                history.write(json.dumps(record, sort_keys=True) + '\n')
        train_text = 'n/a' if train_loss is None else f'{train_loss:.9f}'
        logger.info(f'Validation epoch={epoch} stage={train_stage} train_loss={train_text} val_loss={validation_loss:.9f} lr={learning_rate:.9e} improved={early_stopping.last_improved} counter={early_stopping.counter}/{early_stopping.patience}')

    def train(self) -> None:
        logger.info('>>>>>>> training start <<<<<<<')
        base_path = Path(self.configs.checkpoints)
        path = base_path / self.configs.dataset_name / self.configs.model_name / self.configs.model_id / f'{self.configs.seq_len}_{self.configs.pred_len}' / self.configs.subfolder_train / f'iter{self.configs.itr_i}'
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f'Training iter{self.configs.itr_i} save to: {path}')
        with open(path / 'configs.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self.configs), f, default_flow_style=False)
        if accelerator.is_main_process:
            (path / 'validation_history.jsonl').write_text('', encoding='utf-8')
        accelerator.project_configuration.set_directories(project_dir=path)
        if self.configs.wandb and accelerator.is_main_process or self.configs.sweep:
            import wandb
            run = wandb.init(project='YOUR_PROJECT_NAME', config={'model_name': self.configs.model_name, 'model_id': self.configs.model_id, 'dataset_name': self.configs.dataset_name, 'seq_len': self.configs.seq_len, 'pred_len': self.configs.pred_len, 'learning_rate': self.configs.learning_rate, 'batch_size': self.configs.batch_size}, dir=path)
            if self.configs.sweep:
                self.configs.learning_rate = wandb.config.learning_rate
                self.configs.batch_size = wandb.config.batch_size
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        model_train = self._build_model()
        model_optim = self._select_optimizer(model_train)
        lr_scheduler = self._select_lr_scheduler(model_optim)
        criterion = self._select_criterion()
        if not self.configs.sweep:
            if self.configs.dataset_name == 'P12' and (not self.configs.use_multi_gpu):
                model_train, model_optim = accelerator.prepare(model_train, model_optim)
            else:
                train_loader, vali_loader, model_train, model_optim = accelerator.prepare(train_loader, vali_loader, model_train, model_optim)
            accelerator.register_for_checkpointing(model_optim)
        else:
            model_train, model_optim = accelerator.prepare(model_train, model_optim)
        initial_optimizer_state = model_optim.state_dict()
        initial_scheduler_state = lr_scheduler.state_dict()
        if not self.configs.use_multi_gpu:
            model_train = model_train.to(f'cuda:{self.configs.gpu_id}')
        for train_stage in range(1, self.configs.n_train_stages + 1):
            early_stopping = EarlyStopping(patience=self.configs.patience, verbose=True, min_epochs=self.configs.min_train_epochs)
            logger.info(f'Train stage {train_stage}/{self.configs.n_train_stages} starts.')
            if self.configs.validate_before_training:
                initial_vali_loss = self.vali(model_train=model_train, vali_loader=vali_loader, criterion=criterion, current_epoch=0, train_stage=train_stage)
                early_stopping(initial_vali_loss, model_train, path, current_epoch=0)
                initial_lr = model_optim.param_groups[0]['lr']
                self._record_validation(path=path, train_stage=train_stage, epoch=0, train_loss=None, validation_loss=initial_vali_loss, learning_rate=initial_lr, early_stopping=early_stopping)
                if self.configs.wandb and accelerator.is_main_process or self.configs.sweep:
                    wandb.log({'epoch': 0, 'loss_val': initial_vali_loss, 'learning_rate': initial_lr})
            for epoch_index in tqdm(range(self.configs.train_epochs), desc='Epochs'):
                epoch_number = epoch_index + 1
                train_loss = []
                model_train.train()
                with tqdm(total=len(train_loader), leave=False, desc='Training') as it:
                    batch: dict[str, Tensor]
                    for i, batch in enumerate(train_loader):
                        if batch[next(iter(batch))].shape[0] != self.configs.batch_size and epoch_index == 0:
                            logger.warning(f'Batch No.{i} of total {len(train_loader)} has actual batch_size={batch[next(iter(batch))].shape[0]}, which is not the same as --batch_size={self.configs.batch_size}')
                        if 'y_mask' in batch.keys():
                            if torch.sum(batch['y_mask']).item() == 0:
                                if epoch_index == 0:
                                    logger.warning(f'Batch No.{i} of total {len(train_loader)} has no evaluation point (inferred from y_mask), thus skipping')
                                continue
                        model_optim.zero_grad()
                        if not self.configs.use_multi_gpu:
                            batch = {k: v.to(f'cuda:{self.configs.gpu_id}') for k, v in batch.items()}
                        outputs: dict[str, Tensor] = model_train(exp_stage='train', train_stage=train_stage, current_epoch=epoch_index, **batch)
                        if i == 0 and epoch_index == 0:
                            self._check_model_outputs(batch, outputs)
                        loss: Tensor = criterion(exp_stage='train', model=model_train, **outputs)['loss']
                        if torch.any(torch.isnan(loss)):
                            logger.exception('Loss is nan! Training interruptted!')
                            for key, value in outputs.items():
                                if key == 'loss':
                                    continue
                                elif type(value).__name__ != 'Tensor' and torch.any(torch.isnan(value)):
                                    logger.error(f"Nan value found in model's output tensor '{key}' of shape {value.shape}: {value}")
                            logger.info('Hint: possible cause for nan loss: 1. large learning rate; 2. sqrt(0); 3. ReLU->LeakyReLU')
                            exit(1)
                        train_loss.append(loss.item())
                        if accelerator.is_main_process:
                            it.update()
                            it.set_postfix(loss=f'{loss.item():.2e}')
                        if self.configs.sweep:
                            loss.backward(retain_graph=self.configs.retain_graph)
                        else:
                            accelerator.backward(loss, retain_graph=self.configs.retain_graph)
                        if self.configs.gradient_clip_norm > 0.0:
                            if self.configs.sweep:
                                torch.nn.utils.clip_grad_norm_(model_train.parameters(), self.configs.gradient_clip_norm)
                            else:
                                accelerator.clip_grad_norm_(model_train.parameters(), self.configs.gradient_clip_norm)
                        model_optim.step()
                mean_train_loss = float(np.mean(train_loss))
                vali_loss = self.vali(model_train=model_train, vali_loader=vali_loader, criterion=criterion, current_epoch=epoch_number, train_stage=train_stage)
                current_lr = model_optim.param_groups[0]['lr']
                early_stopping(vali_loss, model_train, path, current_epoch=epoch_number)
                self._record_validation(path=path, train_stage=train_stage, epoch=epoch_number, train_loss=mean_train_loss, validation_loss=vali_loss, learning_rate=current_lr, early_stopping=early_stopping)
                if self.configs.wandb and accelerator.is_main_process or self.configs.sweep:
                    wandb.log({'epoch': epoch_number, 'loss_train': mean_train_loss, 'loss_val': vali_loss, 'learning_rate': current_lr})
                if early_stopping.early_stop:
                    logger.info(f'Early stopping at epoch {epoch_number} (minimum {self.configs.min_train_epochs})')
                    accelerator.set_trigger()
                lr_scheduler.step()
                logger.debug(f'Updating learning rate to {lr_scheduler.get_last_lr()[0]:.6e}')
                if accelerator.check_trigger():
                    accelerator.wait_for_everyone()
                    break
            model_optim.load_state_dict(initial_optimizer_state)
            lr_scheduler.load_state_dict(initial_scheduler_state)

    def test(self) -> None:
        logger.info('>>>>>>> testing start <<<<<<<')
        task_key_mapping = {'short_term_forecast': 'forecasting', 'long_term_forecast': 'forecasting'}
        task_key = task_key_mapping.get(self.configs.task_name, self.configs.task_name)
        if self.configs.test_flop:
            self.configs.batch_size = 1
            logger.info('Batch size overwritten to 1 for FLOPs test.')
            model = self._build_model()
            test_data, test_loader = self._get_data(flag='test')
            batch_dict = next(iter(test_loader))
            real_inputs = {key: value.to(self.device) for key, value in batch_dict.items() if hasattr(value, 'to')}
            logger.info('Using a real data batch (dictionary format) for FLOPs test.')
            test_params_and_flops(model=model, dummy_inputs=real_inputs, model_id=self.configs.model_id, task_key=task_key)
            exit(0)
        if self.configs.test_train_time:
            logger.info(f'Using batch_size={self.configs.batch_size} for training time test.')
            train_data, train_loader = self._get_data(flag='train')
            model = self._build_model()
            criterion = self._select_criterion()
            optimizer = self._select_optimizer(model)
            test_training_step_time(model=model, dataloader=train_loader, criterion=criterion, optimizer=optimizer, model_id=self.configs.model_id, gpu_id=self.configs.gpu_id)
            exit(0)
        if self.configs.test_inference_time:
            logger.info(f'Using batch_size={self.configs.batch_size} for inference time test.')
            test_data, test_loader = self._get_data(flag='test')
            model = self._build_model()
            test_inference_step_time(model=model, dataloader=test_loader, model_id=self.configs.model_id, gpu_id=self.configs.gpu_id)
            exit(0)
        if self.configs.test_gpu_memory:
            logger.info(f'Using batch_size={self.configs.batch_size} for GPU memory test.')
            train_data, train_loader = self._get_data(flag='train')
            batch = next(iter(train_loader))
            batch = {k: v.to(f'cuda:{self.configs.gpu_id}') for k, v in batch.items()}
            model = self._build_model()
            criterion = self._select_criterion()
            test_peak_gpu_memory(model=model, batch=batch, criterion=criterion, model_id=self.configs.model_id, gpu_id=self.configs.gpu_id)
            exit(0)
        if self.configs.test_dataset_statistics:
            _, data_loader = self._get_data(flag='test_all')
            n_observations_raw = 0
            n_observations_all = 0
            logger.info(f"Testing Dataset '{self.configs.dataset_name}':\n            - seq_len={self.configs.seq_len}\n            - pred_len={self.configs.pred_len}\n            - batch_size={self.configs.batch_size}\n            - collate_fn='{self.configs.collate_fn}'")
            logger.warning('Make sure seq_len and pred_len are correctly set.')
            for batch in tqdm(data_loader):
                n_observations_raw += np.sum(batch['x_mask'].detach().cpu().numpy())
                n_observations_raw += np.sum(batch['y_mask'].detach().cpu().numpy())
                n_observations_all += np.sum(np.ones_like(batch['x_mask'].detach().cpu().numpy()))
                n_observations_all += np.sum(np.ones_like(batch['y_mask'].detach().cpu().numpy()))
            logger.info(f'No. observations (raw): {n_observations_raw}')
            logger.info(f'No. observations (all): {n_observations_all}')
            exit(0)
        flag = 'test_all' if self.configs.test_all else 'test'
        test_data, test_loader = self._get_data(flag=flag)
        checkpoint_location: Path = None
        actual_itrs = 1
        if self.configs.checkpoints_test is None:
            base_path = Path(self.configs.checkpoints)
            checkpoint_location = base_path / self.configs.dataset_name / self.configs.model_name / self.configs.model_id / f'{self.configs.seq_len}_{self.configs.pred_len}'
            if self.configs.load_checkpoints_test:
                try:
                    child_folders = [(entry.name, entry) for entry in checkpoint_location.iterdir() if entry.is_dir()]
                    if len(child_folders) == 0:
                        logger.exception(f"No folder under '{checkpoint_location}' matches the model_id '{self.configs.model_id}'.", stack_info=True)
                        logger.exception(f'Tips: Failed to infer the latest checkpoint folder. Please manually provide the checkpoints_test argument pointing to the folder of checkpoint file')
                        exit(1)
                    latest_folder: str = sorted(child_folders, key=lambda item: datetime.datetime.strptime(item[0], '%Y_%m%d_%H%M'))[-1][1].name
                    checkpoint_location = checkpoint_location / latest_folder
                    actual_itrs = len([entry.name for entry in checkpoint_location.iterdir() if entry.is_dir()])
                except Exception as e:
                    logger.exception(f'{e}', stack_info=True)
                    logger.exception(f'Tips: Failed to infer the latest checkpoint folder. Please manually provide the checkpoints_test argument pointing to the folder of checkpoint file')
                    exit(1)
            else:
                train_folder = datetime.datetime.now().strftime('%Y_%m%d_%H%M')
                path = checkpoint_location / train_folder / f'iter0'
                path.mkdir(parents=True, exist_ok=True)
                checkpoint_location = checkpoint_location / train_folder
        for itr_i in range(actual_itrs):
            if self.configs.checkpoints_test is None:
                checkpoint_location_itr = checkpoint_location / f'iter{itr_i}'
            else:
                checkpoint_location_itr = Path(self.configs.checkpoints_test)
            model_test = self._build_model().eval()
            if self.configs.load_checkpoints_test:
                checkpoint_file = checkpoint_location_itr / 'pytorch_model.bin'
                if checkpoint_file.exists():
                    try:
                        original_state_dict = self._get_state_dict(checkpoint_file)
                        load_result = model_test.load_state_dict(original_state_dict, strict=False)
                        if load_result.missing_keys or load_result.unexpected_keys:
                            logger.warning(f'The following keys in checkpoint are not correctly loaded:\n                            load_result.missing_keys={load_result.missing_keys!r}\n                            load_result.unexpected_keys={load_result.unexpected_keys!r}\n\n                            Results may be incorrect!\n                            ')
                    except Exception as e:
                        logger.exception(f'{e}', stack_info=True)
                        logger.exception(f'Failed to load checkpoint file at {checkpoint_file}. Skipping it...')
                        continue
                else:
                    try:
                        load_checkpoint_in_model(model_test, checkpoint_location_itr)
                    except Exception as e:
                        logger.exception(f'{e}', stack_info=True)
                        logger.exception(f'Failed to load checkpoint file at {checkpoint_file}. Skipping it...')
                        continue
            model_test, test_loader = accelerator.prepare(model_test, test_loader)
            if not self.configs.use_multi_gpu:
                model_test = model_test.to(f'cuda:{self.configs.gpu_id}')
            subfolder_eval = f"eval_{datetime.datetime.now().strftime('%Y_%m%d_%H%M')}"
            folder_path = checkpoint_location_itr / subfolder_eval
            folder_path.mkdir(exist_ok=True)
            logger.info(f'Testing results will be saved under {folder_path}')
            array_dict = {}
            if self.configs.task_name in ['short_term_forecast', 'long_term_forecast', 'imputation']:
                input_tensor_names = ['x', 'y', 'x_mask', 'y_mask', 'sample_ID']
                output_tensor_names = ['pred']
            else:
                raise NotImplementedError
            for tensor_name in input_tensor_names + output_tensor_names:
                array_dict[tensor_name] = []
            with torch.no_grad():
                batch: dict[str, Tensor]
                for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), leave=False, desc='Testing'):
                    if batch[next(iter(batch))].shape[0] != self.configs.batch_size:
                        logger.warning(f'Batch No.{i} of total {len(test_loader)} has actual batch_size={batch[next(iter(batch))].shape[0]}, which is not the same as --batch_size={self.configs.batch_size}')
                    if not self.configs.use_multi_gpu:
                        batch = {k: v.to(f'cuda:{self.configs.gpu_id}') for k, v in batch.items()}
                    outputs: dict[str, Tensor] = model_test(exp_stage='test', **batch)
                    if i == 0 and itr_i == 0:
                        self._check_model_outputs(batch, outputs)
                    batch_all: list[dict] = accelerator.gather_for_metrics([batch])
                    batch_all: dict = self._merge_gathered_dicts(batch_all)
                    outputs_all: list[dict] = accelerator.gather_for_metrics([outputs])
                    outputs_all: dict = self._merge_gathered_dicts(outputs_all)
                    for tensor_name in input_tensor_names:
                        if tensor_name in batch_all.keys():
                            array_dict[tensor_name].append(batch_all[tensor_name].detach().cpu().numpy())
                    for tensor_name in output_tensor_names:
                        if tensor_name in outputs_all.keys():
                            array_dict[tensor_name].append(outputs_all[tensor_name].detach().cpu().numpy())
            for tensor_name in input_tensor_names + output_tensor_names:
                if len(array_dict[tensor_name]) > 0:
                    array_dict[tensor_name] = self._concatenate_test_batches(array_dict[tensor_name])
                else:
                    array_dict[tensor_name] = None
            metrics = None
            if self.configs.task_name in ['short_term_forecast', 'long_term_forecast', 'imputation']:
                metrics = metric(**array_dict)
                if self.configs.dataset_name == 'P12':
                    metrics['N_eval'] = int(array_dict['y_mask'].sum())
                if self.configs.wandb and accelerator.is_main_process and self.configs.is_training or self.configs.sweep:
                    import wandb
                    wandb.log({'loss_test': np.mean(metrics['MSE'])})
            if metrics is not None:
                for key, value in metrics.items():
                    if isinstance(value, np.float32):
                        metrics[key] = float(value)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, np.float32):
                                metrics[key] = [float(v) for v in value]
                                break
                logger.info('Metrics:\n%s', json.dumps(metrics, indent=4))
                with open(folder_path / 'metric.json', 'w') as f:
                    json.dump(metrics, f, indent=2)
            if self.configs.save_arrays:
                for tensor_name in input_tensor_names:
                    if array_dict[tensor_name] is not None:
                        np.save(folder_path / f'input_{tensor_name}.npy', array_dict[tensor_name])
                for tensor_name in output_tensor_names:
                    if array_dict[tensor_name] is not None:
                        np.save(folder_path / f'output_{tensor_name}.npy', array_dict[tensor_name])
