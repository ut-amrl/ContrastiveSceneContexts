
from model.cluster_label_model import ClusterLabelModel
import torch
from omegaconf import OmegaConf
import hydra
import os
import logging

import lib.unoriginal.multiprocessing_utils as mpu
import numpy as np
import MinkowskiEngine as ME

from lib.test_data_loader import createDataLoader, TestDataset
from lib.train_data_loader import createTrainingDataLoader, TrainDataset
from lib.contrastive_trainer import NPairLossClusterTrainer, TripletLossTrainer

torch.manual_seed(0)
torch.cuda.manual_seed(0)

@hydra.main(config_path='config', config_name='defaults.yaml')
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
  # trainingDataLoader = createTrainingDataLoader("/home/amanda/Downloads/testFile2.csv", config.data.voxel_size, config.data.batch_size)



#
# def test(testFiles, model, config):
#   dataLoader =  createDataLoader(testFiles, config.data.voxel_size, config.data.batch_size)
#   model.eval()
#   output = []
#
#   with torch.no_grad():
#     for batch in dataLoader:
#       coords = batch[0]
#       feats = batch[1]
#       tensor = ME.SparseTensor(coords=coords, feats=feats)
#
#       outForBatch = model(tensor)
#       output.append(outForBatch)
#   print("Output")
#   print(output[0].F)

def testTrainer(model, config, datasetTopFile):
  trainingDataLoader = createTrainingDataLoader(datasetTopFile, config.data.voxel_size, config.data.batch_size, config.misc.num_gpus)
  if (config.trainer.useNPairLoss):
    trainer = NPairLossClusterTrainer(model, config, trainingDataLoader)
  else:
    trainer = TripletLossTrainer(model, config, trainingDataLoader)
  trainer.train()


if __name__ == "__main__":
  os.environ['MKL_THREADING_LAYER'] = 'GNU'
  main()
