
import logging
import os

import torch
import torch.optim as optim
from torch.serialization import default_restore_location
from trainer_base import ClusterTrainer

import MinkowskiEngine as ME

from tensorboardX import SummaryWriter
from omegaconf import OmegaConf

from model.cluster_label_model import ClusterLabelModel
import lib.unoriginal.distributed as du
from lib.unoriginal.timer import Timer, AverageMeter

def load_state(model, weights, lenient_weight_loading=False):
  if du.get_world_size() > 1:
      _model = model.module
  else:
      _model = model

  if lenient_weight_loading:
    model_state = _model.state_dict()
    filtered_weights = {
        k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
    }
    logging.info("Load weights:" + ', '.join(filtered_weights.keys()))
    weights = model_state
    weights.update(filtered_weights)

  _model.load_state_dict(weights, strict=True)

class ClusterClassificationLossTrainer(ClusterTrainer):


    def __init__(self, initial_model, config, data_loader):

        super(ClusterTrainer, self).__init__(initial_model, config, data_loader)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)


    def trainIter(self, data_loader_iter, timers):
        # print("HERE!")
        data_meter, data_timer, total_timer = timers
        self.optimizer.zero_grad()
        batch_loss = {
            'loss': 0.0,
        }
        data_time = 0
        total_timer.tic()

        data_timer.tic()
        coords, feats, labels = data_loader_iter.next()
        data_time += data_timer.toc(average=False)
        modelInput = ME.SparseTensor(feats=feats.to(self.cur_device), coords=coords.to(self.cur_device))
        self.model = self.model.double()
        modelOut = self.model(modelInput)

        loss = self.criterion(modelOut.F.squeeze(), labels.long())
        loss.backward()

        result = {"loss": loss}
        if self.config.misc.num_gpus > 1:
            result = du.scaled_all_reduce_dict(result, self.config.misc.num_gpus)
        batch_loss['loss'] += result["loss"].item()

        self.optimizer.step()


        torch.cuda.empty_cache()
        total_timer.toc()
        data_meter.update(data_time)
        return batch_loss


