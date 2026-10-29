import gc
import logging
import resource
import time
from typing import Dict, List
from tqdm import tqdm

import numpy as np
import torch
import torch.optim
from torch.utils.data import DataLoader

from models import BaseModel
from utils import const, utils

from .dataset import *


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch',
                            type=int,
                            default=1,
                            help='Number of epochs.')
        parser.add_argument('--lr',
                            type=float,
                            default=1e-3,
                            help='Learning rate.')
        parser.add_argument(
            '--patience',
            type=int,
            default=3,
            help=
            'Number of epochs with no improvement after which learning rate will be reduced'
        )
        parser.add_argument(
            '--early_stop',
            type=int,
            default=5,
            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--min_lr', type=float, default=1e-6)
        parser.add_argument('--l2',
                            type=float,
                            default=3e-5,
                            help='Weight decay in optimizer.')

        parser.add_argument('--infoNCE_neg_sample', type=int, default=1024)

        parser.add_argument('--batch_size',
                            type=int,
                            default=1024,
                            help='Batch size during training.')
        parser.add_argument(
            '--src_batch_size',
            type=int,
            default=0,
            help=('Search-stage batch size for rec_then_src training; '
                  '0 reuses --batch_size.'))
        parser.add_argument('--eval_batch_size',
                            type=int,
                            default=512,
                            help='Batch size during testing.')
        parser.add_argument(
            '--num_workers',
            type=int,
            default=10,
            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument(
            '--max_train_steps',
            type=int,
            default=0,
            help='Limit batches per epoch for smoke tests; 0 is unlimited.')
        parser.add_argument(
            '--max_eval_batches',
            type=int,
            default=0,
            help='Limit eval batches for smoke tests; 0 is unlimited.')

        return parser

    def __init__(self, args) -> None:

        self.epoch = args.epoch
        self.print_interval = 500

        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.patience = args.patience
        self.min_lr = args.min_lr
        self.l2 = args.l2
        self.batch_size = args.batch_size
        self.src_batch_size = args.src_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.num_workers = args.num_workers

        self.infoNCE_neg_sample = args.infoNCE_neg_sample
        self.max_train_steps = args.max_train_steps
        self.max_eval_batches = args.max_eval_batches

        self.topk = [1, 5, 10]
        self.metrics = ['NDCG', 'HR']
        # early stop based on main_metric
        self.main_metric = 'NDCG@10'

        self.train_loader: DataLoader = None
        self.val_loader: DataLoader = None
        self.optimizer: torch.optim.Optimizer = None

        self.datasetParaDict = {
            "user_vocab": None,
        }
        self.query_vocab = None

    def _build_optimizer(self, model: BaseModel):
        self.optimizer = torch.optim.Adam(model.customize_parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.l2)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=self.patience,
            min_lr=self.min_lr,)
            #verbose=True

    def getDataLoader(self, dataset: BaseDataSet, batch_size: int,
                      shuffle: bool) -> DataLoader:
        kwargs = {
            'dataset': dataset,
            'shuffle': shuffle,
            'batch_size': batch_size,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'collate_fn': dataset.collate_batch,
        }
        # PyTorch rejects prefetch_factor and persistent_workers when no
        # worker processes are used; num_workers=0 is the requested safe mode.
        if self.num_workers > 0:
            kwargs.update(
                prefetch_factor=batch_size // self.num_workers + 1,
                worker_init_fn=utils.worker_init_fn,
                persistent_workers=True,
            )
        return DataLoader(**kwargs)

    def set_dataloader(self):
        raise NotImplementedError

    def train(self, model: BaseModel):
        self._build_optimizer(model)

        if model.query_item_alignment:
            self.get_query_vocab()
            self.InfoNCE_dataloader = self.getDataLoader(
                InfoNCEDataset(query_vocab=self.query_vocab),
                self.infoNCE_neg_sample,
                shuffle=False)

        main_metric_results, dev_results = list(), list()
        for epoch in range(self.epoch):
            gc.collect()
            torch.cuda.empty_cache()

            epoch_loss = self.train_epoch(epoch, model)

            logging.info("epoch:{} mean loss:{:.4f}".format(epoch, epoch_loss))

            dev_result, main_result = self.evaluate(model, 'val')
            dev_results.append(dev_result)
            main_metric_results.append(main_result)
            logging.info("Dev Result:")
            logging.info(dev_result)
            print("Dev Result: {}".format(dev_result))
            self.scheduler.step(main_result)

            if max(main_metric_results) == main_metric_results[-1]:
                model.save_model()

            if self.early_stop > 0 and self.eval_termination(
                    main_metric_results):
                logging.info('Early stop at %d based on dev result.' %
                             (epoch + 1))
                break

        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(" ")
        logging.info("Best Dev Result at epoch:{}".format(best_epoch))
        logging.info(dev_results[best_epoch])
        print("\nBest Dev Result at epoch: {}\n {}".format(
            best_epoch, dev_results[best_epoch]))

        model.load_model()
        return {'best_epoch': best_epoch,
                'validation': dev_results[best_epoch]}

    def train_epoch(self, epoch: int, model: BaseModel):
        raise NotImplementedError

    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False

    @staticmethod
    def evaluate_method(predictions: np.ndarray, topk: int,
                        metrics: List[str]) -> Dict[str, float]:
        if predictions.ndim != 2 or predictions.shape[1] < 1:
            raise ValueError('predictions must have shape [rows, candidates]')
        evaluations = dict()
        positive = predictions[:, :1]
        greater = (predictions[:, 1:] > positive).sum(axis=1)
        tied = (predictions[:, 1:] == positive).sum(axis=1) + 1
        tie_offsets = np.arange(1, predictions.shape[1] + 1)[None, :]
        tie_positions = greater[:, None] + tie_offsets
        belongs_to_tie = tie_offsets <= tied[:, None]
        for k in topk:
            visible_ties = np.clip(k - greater, 0, tied)
            expected_hit = visible_ties / tied
            expected_ndcg = (
                belongs_to_tie
                & (tie_positions <= k)
            ) / np.log2(tie_positions + 1)
            expected_ndcg = expected_ndcg.sum(axis=1) / tied
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'HR':
                    evaluations[key] = expected_hit.mean()
                elif metric == 'NDCG':
                    evaluations[key] = expected_ndcg.mean()
                else:
                    raise ValueError(
                        'Undefined evaluation metric: {}.'.format(metric))
        return evaluations

    @torch.no_grad()
    def predict(self, model: BaseModel, data_loader: DataLoader):
        model.eval()
        predictions = list()

        start = time.time()
        for step, batch in enumerate(data_loader):
            prediction = model.predict(utils.batch_to_gpu(batch, model.device))
            predictions.extend(prediction.cpu().data.numpy())
            if self.max_eval_batches > 0 and step + 1 >= self.max_eval_batches:
                break

        logging.info("model evaluate time used:{}s".format(time.time() -
                                                           start))
        predictions = np.array(predictions)

        return predictions

    def evaluate(self, model: BaseModel, mode: str):
        raise NotImplementedError

    def build_dataset(self):
        if self.datasetParaDict['user_vocab'] is None:
            self.datasetParaDict['user_vocab'] = utils.load_pickle(
                const.user_vocab)

    def get_query_vocab(self):
        if self.query_vocab is None:
            self.query_vocab = utils.load_pickle(const.query_vocab)


class SarRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--src_loss_weight', type=float, default=0.1)
        parser.add_argument('--train_schedule', choices=('joint', 'rec_then_src'),
                            default='joint')
        parser.add_argument('--pretrain_epoch', type=int, default=100)
        parser.add_argument('--finetune_epoch', type=int, default=100)

        return BaseRunner.parse_runner_args(parser)

    def __init__(self, args) -> None:
        super().__init__(args)
        self.train_schedule = args.train_schedule
        self.pretrain_epoch = args.pretrain_epoch
        self.finetune_epoch = args.finetune_epoch
        self.src_loss_weight = args.src_loss_weight
        self.session_items = None
        if (getattr(args, 'query_backend', None) == 'precomputed_qwen'
                and getattr(args, 'session_pos_items_path', '')):
            self.session_items = np.load(args.session_pos_items_path,
                                         mmap_mode='r', allow_pickle=False)

        self.build_dataset()
        self.set_dataloader()

    def set_dataloader(self):
        self.rec_train_loader = self.getDataLoader(self.traindata['rec'],
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        self.rec_val_loader = self.getDataLoader(
            self.valdata['rec'],
            batch_size=self.eval_batch_size,
            shuffle=False)

        if self.train_schedule == 'rec_then_src':
            src_train_batch_size = (self.src_batch_size
                                    if self.src_batch_size > 0
                                    else self.batch_size)
        else:
            src_train_batch_size = len(self.traindata['src']) // (
                len(self.traindata['rec']) // self.batch_size + 1) + 1
        logging.info('src train batch size:{}'.format(src_train_batch_size))
        self.src_train_loader = self.getDataLoader(
            self.traindata['src'],
            batch_size=src_train_batch_size,
            shuffle=True)
        self.src_val_loader = self.getDataLoader(
            self.valdata['src'],
            batch_size=self.eval_batch_size,
            shuffle=False)

    def build_dataset(self):
        super().build_dataset()
        self.traindata = {
            "rec": RecDataSet(
                train='train', user_vocab=self.datasetParaDict['user_vocab'],
                session_items=self.session_items),
            "src": SrcDataSet(
                train='train', user_vocab=self.datasetParaDict['user_vocab'],
                session_items=self.session_items),
        }
        self.valdata = {
            "rec": RecDataSet(
                train='val', user_vocab=self.datasetParaDict['user_vocab'],
                session_items=self.session_items),
            "src": SrcDataSet(
                train='val', user_vocab=self.datasetParaDict['user_vocab'],
                session_items=self.session_items),
        }

    def train(self, model: BaseModel):
        if self.train_schedule == 'joint':
            return super().train(model)
        return self.train_rec_then_src(model)

    def train_task_epoch(self, epoch: int, model: BaseModel, task: str):
        model.train()
        loader = self.rec_train_loader if task == 'rec' else self.src_train_loader
        loss_fn = model.rec_loss if task == 'rec' else model.src_loss
        logging.info(" ")
        logging.info("%s Epoch: %d", task, epoch)
        print("\n{} Epoch: {}".format(task, epoch))

        loss_list = []
        start = time.time()
        for step, batch in enumerate(tqdm(loader)):
            losses = loss_fn(utils.batch_to_gpu(batch, model.device))
            total_loss = losses['total_loss']
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            loss_list.append(total_loss.item())
            if self.max_train_steps > 0 and step + 1 >= self.max_train_steps:
                break
            if step > 0 and step % self.print_interval == 0:
                logging.info(
                    "%s epoch:%d step:%d time:%.2fs total_loss:%.4f",
                    task, epoch, step, time.time() - start,
                    np.mean(loss_list).item())
        logging.info("%s total time: %.2fs", task, time.time() - start)
        return np.mean(loss_list).item()

    def evaluate_task(self, model: BaseModel, mode: str, task: str):
        if mode != 'val':
            raise ValueError('training may evaluate validation data only')
        loaders = {'rec': self.rec_val_loader, 'src': self.src_val_loader}
        predictions = self.predict(model, loaders[task])
        metrics = self.evaluate_method(
            predictions, self.topk, self.metrics)
        return utils.format_metric(metrics), metrics[self.main_metric]

    def run_stage(self, model: BaseModel, stage_name: str, task: str,
                  epochs: int, checkpoint_path: str):
        self._build_optimizer(model)
        main_results = []
        dev_results = []
        for epoch in range(epochs):
            gc.collect()
            torch.cuda.empty_cache()
            epoch_loss = self.train_task_epoch(epoch, model, task)
            logging.info("%s epoch:%d mean loss:%.4f",
                         stage_name, epoch, epoch_loss)
            dev_result, main_result = self.evaluate_task(model, 'val', task)
            main_results.append(main_result)
            dev_results.append(dev_result)
            logging.info("%s Dev Result: %s", stage_name, dev_result)
            print("{} Dev Result: {}".format(stage_name, dev_result))
            self.scheduler.step(main_result)
            if max(main_results) == main_results[-1]:
                model.save_model(model_path=checkpoint_path)
            if self.early_stop > 0 and self.eval_termination(main_results):
                logging.info(
                    "Early stop %s at %d based on %s.",
                    stage_name, epoch + 1, self.main_metric)
                break

        if not main_results:
            raise RuntimeError("No epochs were executed for " + stage_name)
        best_epoch = main_results.index(max(main_results))
        logging.info("Best %s epoch: %d", stage_name, best_epoch)
        model.load_model(model_path=checkpoint_path)
        return {'best_epoch': best_epoch,
                'validation': dev_results[best_epoch]}

    def train_rec_then_src(self, model: BaseModel):
        pretrain_path = model.model_path + '/rec_pretrain_best.pt'
        final_path = model.model_path + '/best.pt'
        logging.info("Starting KuaiSAR recommendation pre-training")
        pretrain = self.run_stage(model, 'rec_pretrain', 'rec',
                                  self.pretrain_epoch, pretrain_path)

        logging.info("Starting KuaiSAR search fine-tuning")
        finetune = self.run_stage(model, 'src_finetune', 'src',
                                  self.finetune_epoch, final_path)
        return {'rec_pretrain': pretrain, 'src_finetune': finetune}

    def train_epoch(self, epoch: int, model: BaseModel):
        model.train()
        logging.info(" ")
        logging.info("Epoch: {}".format(epoch))
        print("\nEpoch: {}".format(epoch))

        if model.query_item_alignment:
            InfoNCE_iterator = iter(self.InfoNCE_dataloader)

        src_iterator = iter(self.src_train_loader)

        loss_list = []
        loss_dict = {"rec": {}, "src": {}}
        start = time.time()
        for step, rec_batch in enumerate(tqdm(self.rec_train_loader)):
            try:
                src_batch = next(src_iterator)
            except StopIteration:
                src_iterator = iter(self.src_train_loader)
                src_batch = next(src_iterator)

            if model.query_item_alignment:
                align_dict = next(InfoNCE_iterator)
                rec_batch.update(align_dict)
                src_batch.update(align_dict)

            rec_loss = model.loss(utils.batch_to_gpu(rec_batch, model.device))
            src_loss = model.src_loss(
                utils.batch_to_gpu(src_batch, model.device))

            for k in rec_loss.keys():
                if k in loss_dict['rec'].keys():
                    loss_dict['rec'][k].append(rec_loss[k].item())
                    loss_dict['src'][k].append(src_loss[k].item())
                else:
                    loss_dict['rec'][k] = [rec_loss[k].item()]
                    loss_dict['src'][k] = [src_loss[k].item()]

            total_loss = rec_loss['total_loss'] + \
                src_loss['total_loss'] * self.src_loss_weight

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            loss_list.append(total_loss.item())
            if self.max_train_steps > 0 and step + 1 >= self.max_train_steps:
                break

            if step > 0 and step % self.print_interval == 0:
                logging.info(
                    "epoch:{:d} step:{:d} time:{:.2f}s |rec {}| |src {}|".
                    format(
                        epoch, step,
                        time.time() - start, " ".join([
                            "{}:{:.4f}".format(k,
                                               np.mean(v).item())
                            for k, v in loss_dict['rec'].items()
                        ]), " ".join([
                            "{}:{:.4f}".format(k,
                                               np.mean(v).item())
                            for k, v in loss_dict['src'].items()
                        ])))
        wall_seconds = time.time() - start
        rec_total = float(np.mean(loss_dict['rec']['total_loss']))
        src_total = float(np.mean(loss_dict['src']['total_loss']))
        joint_total = float(np.mean(loss_list))
        finite = bool(np.isfinite(rec_total) and np.isfinite(src_total)
                      and np.isfinite(joint_total))
        if torch.cuda.is_available():
            cuda_max_allocated = torch.cuda.max_memory_allocated()
            cuda_max_reserved = torch.cuda.max_memory_reserved()
        else:
            cuda_max_allocated = cuda_max_reserved = 0
        logging.info(
            'EPOCH_TRAIN_SUMMARY epoch=%d rec_total_loss=%.8f '
            'src_total_loss=%.8f joint_total_loss=%.8f lr=%.10g '
            'wall_seconds=%.3f train_batches=%d finite=%s max_rss_bytes=%d '
            'cuda_max_allocated=%d cuda_max_reserved=%d',
            epoch, rec_total, src_total, joint_total,
            self.optimizer.param_groups[0]['lr'], wall_seconds, len(loss_list),
            finite, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            cuda_max_allocated, cuda_max_reserved)
        logging.info("total time: {:.2f}s".format(wall_seconds))

        return joint_total

    def evaluate(self, model: BaseModel, mode: str):
        if mode != 'val':
            raise ValueError('training may evaluate validation data only')
        rec_predictions = self.predict(model, self.rec_val_loader)
        src_predictions = self.predict(model, self.src_val_loader)
        rec_results = self.evaluate_method(rec_predictions, self.topk,
                                           self.metrics)
        src_results = self.evaluate_method(src_predictions, self.topk,
                                           self.metrics)

        results = {
            "rec": utils.format_metric(rec_results),
            "src": utils.format_metric(src_results)
        }
        logging.info(
            'VALIDATION_SUMMARY rec_samples=%d src_samples=%d '
            'rec_candidates=%d src_candidates=%d full_coverage=%s '
            'loader_shuffle=false max_eval_batches=%d',
            len(rec_predictions), len(src_predictions),
            rec_predictions.shape[1], src_predictions.shape[1],
            self.max_eval_batches == 0, self.max_eval_batches)
        return results, (rec_results[self.main_metric] +
                         src_results[self.main_metric]) / 2.0
