
from model.cluster_label_model import ClusterLabelModel
import torch
from omegaconf import OmegaConf
import hydra
import os
import logging

import lib.unoriginal.multiprocessing_utils as mpu
import numpy as np
import MinkowskiEngine as ME

from lib.classification_data_loader import createClassificationDataLoader
from lib.classification_trainer import ClusterClassificationLossTrainer

torch.manual_seed(0)
torch.cuda.manual_seed(0)

@hydra.main(config_path='config', config_name='classification.yaml')
def main(config):
  if os.path.exists('config.yaml'):
    logging.info('===> Loading exsiting config file')
    config = OmegaConf.load('config.yaml')
    logging.info('===> Loaded exsiting config file')
  logging.info('===> Configurations')
  #logging.info(config.pretty())

  if config.misc.num_gpus > 1:
      mpu.multi_proc_run(config.misc.num_gpus, port=config.misc.port,
              fun=single_proc_run, fun_args=(config,))
  else:
      single_proc_run(config)

def single_proc_run(config):
  num_feats=1 # LiDAR just has reflectance/intensity
  model = ClusterLabelModel(num_feats, config.net.model_n_out, config, D=3)
  model = model.double()
  print(model)
  model.updateWithPretrainedWeights(config.net.pretrained_weights)
  testTrainer(model, config, config.data.dataset_file)


def testTrainer(model, config, datasetTopFile):
  trainer = ClusterClassificationLossTrainer(model, config, createClassificationDataLoader(datasetTopFile, config.data.voxel_size, config.data.batch_size, config.misc.num_gpus))
  trainer.train()


if __name__ == "__main__":
  os.environ['MKL_THREADING_LAYER'] = 'GNU'
  main()
