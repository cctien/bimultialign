import logging
import math
import time
from typing import Any, Dict, Iterator, Optional, Union

from allennlp.common import Tqdm
from allennlp.common import util as common_util
from allennlp.data.dataloader import TensorDict
from allennlp.nn import util as nn_util
from allennlp.training import Trainer, GradientDescentTrainer
from allennlp.training import util as training_util
from allennlp.training.optimizers import Optimizer
from overrides import overrides
import torch
from torch.cuda import amp
import torch.distributed as dist

logger = logging.getLogger(__name__)


@Trainer.register('gan', constructor='from_partial_objects')
@Trainer.register('multiplayer', constructor='from_partial_objects')
class MultiplayerTrainer(GradientDescentTrainer):
    """
    """

    @overrides
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        cpu_memory_usage = []
        for worker, memory in common_util.peak_cpu_memory().items():
            cpu_memory_usage.append((worker, memory))
            logger.info(f"Worker {worker} memory usage: "
                        f"{common_util.format_size(memory)}")
        gpu_memory_usage = []
        for gpu, memory in common_util.peak_gpu_memory().items():
            gpu_memory_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage: "
                        f"{common_util.format_size(memory)}")

        regularization_penalty = self.model.get_regularization_penalty()

        train_loss = 0.0
        batch_loss = 0.0
        train_reg_loss = None if regularization_penalty is None else 0.0
        batch_reg_loss = None if regularization_penalty is None else 0.0

        # Set the model to "train" mode.
        self._pytorch_model.train()

        # Get tqdm for the training batches
        batch_generator = iter(self.data_loader)
        batch_group_generator = common_util.lazy_groups_of(batch_generator,
                                                           self._num_gradient_accumulation_steps)

        logger.info("Training")

        num_training_batches: Union[int, float]
        try:
            len_data_loader = len(self.data_loader)
            num_training_batches = math.ceil(len_data_loader
                                             / self._num_gradient_accumulation_steps)
        except TypeError:
            num_training_batches = float('inf')

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the master's
        # progress is shown
        if self._master:
            batch_group_generator_tqdm = Tqdm.tqdm(batch_group_generator,
                                                   total=num_training_batches)
        else:
            batch_group_generator_tqdm = batch_group_generator

        self._last_log = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        done_early = False
        for batch_group in batch_group_generator_tqdm:
            if self._distributed:
                # Check whether the other workers have stopped already (due to differing amounts of
                # data in each). If so, we can't proceed because we would hang when we hit the
                # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
                # here because NCCL process groups apparently don't support BoolTensor.
                done = torch.as_tensor(0, device=self.cuda_device)
                torch.distributed.all_reduce(done,
                                             torch.distributed.ReduceOp.SUM)
                if done.item() > 0:
                    done_early = True
                    logger.warning(f"Worker {torch.distributed.get_rank()} finishing training early! "
                                   "This implies that there is an imbalance in your training "
                                   "data across the workers and that some amount of it will be "
                                   "ignored. A small amount of this is fine, but a major imbalance "
                                   "should be avoided. Note: This warning will appear unless your "
                                   "data is perfectly balanced.")
                    break

            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            # ====
            # Backpropagation starts.
            # ====

            for _player, _num_steps in self.optimizer.nums_steps.items():
                _optimizer = self.optimizer.optimizers[_player]
                for _ in range(_num_steps):
                    batch_output_dict = self._step_batch(player=_player,
                                                         optimizer=_optimizer,
                                                         batch_group=batch_group,
                                                         batch_num_total=batch_num_total,
                                                         train_loss=train_loss,
                                                         train_reg_loss=train_reg_loss)

            train_loss = batch_output_dict['train_loss']
            batch_loss = batch_output_dict['batch_loss']
            train_reg_loss = batch_output_dict['train_reg_loss']
            batch_reg_loss = batch_output_dict['batch_reg_loss']
            batch_group_outputs = batch_output_dict['batch_group_outputs']
            batch_grad_norm = batch_output_dict['batch_grad_norm']
            param_updates = batch_output_dict['param_updates']

            # ====
            # Backpropagation ends.
            # ====

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(self.model,
                                                train_loss,
                                                train_reg_loss,
                                                batch_loss,
                                                batch_reg_loss,
                                                batches_this_epoch,
                                                world_size=self._world_size,
                                                cuda_device=self.cuda_device)

            if self._master:
                # Updating tqdm only for the master as the trainers wouldn't have one
                description = training_util.description_from_metrics(metrics)
                batch_group_generator_tqdm.set_description(description,
                                                           refresh=False)
                self._tensorboard.log_batch(self.model,
                                            self.optimizer,
                                            batch_grad_norm,
                                            metrics,
                                            batch_group,
                                            param_updates)

                self._checkpointer.maybe_save_checkpoint(self,
                                                         epoch, batches_this_epoch)

            for callback in self._batch_callbacks:
                callback(self,
                         batch_group,
                         batch_group_outputs,
                         epoch,
                         batches_this_epoch,
                         is_training=True,
                         is_master=self._master)

        if self._distributed and not done_early:
            logger.warning(f"Worker {torch.distributed.get_rank()} "
                           "completed its entire epoch (training).")
            # Indicate that we're done so that any workers that have remaining data stop the epoch early.
            done = torch.as_tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        # Let all workers finish their epoch before computing
        # the final statistics for the epoch.
        if self._distributed:
            dist.barrier()

        metrics = training_util.get_metrics(self.model,
                                            train_loss,
                                            train_reg_loss,
                                            batch_loss=None,
                                            batch_reg_loss=None,
                                            num_batches=batches_this_epoch,
                                            reset=True,
                                            world_size=self._world_size,
                                            cuda_device=self.cuda_device)

        for (worker, memory) in cpu_memory_usage:
            metrics["worker_" + str(worker) + "_memory_MB"] = memory

        for (gpu_num, memory) in gpu_memory_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory

        return metrics

    def _step_batch(self,
                    player: str,
                    optimizer: Optimizer,
                    batch_group: Iterator[TensorDict],
                    batch_num_total: int,
                    train_loss: float,
                    train_reg_loss: Optional[float],
                    ) -> Dict[str, Any]:

        optimizer.zero_grad()

        batch_group_outputs = []
        for batch in batch_group:
            with amp.autocast(self._use_amp):
                batch_outputs = self.batch_outputs(batch,
                                                   for_training=True,
                                                   player=player)
                batch_group_outputs.append(batch_outputs)

                loss = batch_outputs.get('loss')
                reg_loss = batch_outputs.get('reg_loss')

                if torch.isnan(loss):
                    raise ValueError("nan loss encountered")

                loss = loss / len(batch_group)
                batch_loss = loss.item()
                train_loss += batch_loss
                if reg_loss is not None:
                    reg_loss = reg_loss / len(batch_group)
                    batch_reg_loss = reg_loss.item()
                    train_reg_loss += batch_reg_loss
                else:
                    batch_reg_loss = None

            if self._scaler is not None:
                self._scaler.scale(loss).backward()
            else:
                loss.backward()

        batch_grad_norm = self.rescale_gradients()

        # This does nothing if batch_num_total is None or you are using a
        # scheduler which doesn't update per batch.
        if self._learning_rate_scheduler:
            self._learning_rate_scheduler.step_batch(batch_num_total)

        if self._momentum_scheduler:
            self._momentum_scheduler.step_batch(batch_num_total)

        param_updates = None

        if self._tensorboard.should_log_histograms_this_batch() and self._master:
            # Get the magnitude of parameter updates for logging.  We need to do some
            # computation before and after the optimizer step, and it's expensive because of
            # GPU/CPU copies (necessary for large models, and for shipping to tensorboard), so
            # we don't do this every batch, only when it's requested.
            param_updates = {name: param.detach().cpu().clone()
                             for name, param in self.model.named_parameters()}

            if self._scaler is not None:
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                optimizer.step()

            for name, param in self.model.named_parameters():
                param_updates[name].sub_(param.detach().cpu())
        else:
            if self._scaler is not None:
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                optimizer.step()

        return {'batch_group_outputs': batch_group_outputs,
                'batch_grad_norm': batch_grad_norm,
                'param_updates': param_updates,
                'batch_loss': batch_loss,
                'train_loss': train_loss,
                'batch_reg_loss': batch_reg_loss,
                'train_reg_loss': train_reg_loss, }

    @overrides
    def batch_outputs(self,
                      batch: TensorDict,
                      for_training: bool,
                      player: Optional[str] = None,
                      ) -> Dict[str, torch.Tensor]:
        """
        Does a forward pass on the given batch and returns the output dictionary that the model
        returns, after adding any specified regularization penalty to the loss (if training).
        """
        batch = nn_util.move_to_device(batch, self.cuda_device)
        output_dict = self._pytorch_model(**batch, player=player)

        if for_training:
            try:
                assert 'loss' in output_dict
                regularization_penalty = self.model.get_regularization_penalty()

                if regularization_penalty is not None:
                    output_dict['reg_loss'] = regularization_penalty
                    output_dict['loss'] += regularization_penalty

            except AssertionError:
                if for_training:
                    raise RuntimeError("The model you are trying to optimize does not contain a"
                                       " 'loss' key in the output of model.forward(inputs).")

        return output_dict
