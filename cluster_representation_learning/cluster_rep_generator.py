
from model.cluster_label_model import ClusterLabelModel
import torch
from omegaconf import OmegaConf
import hydra
import os
import logging
import joblib

import lib.unoriginal.multiprocessing_utils as mpu
import numpy as np
import MinkowskiEngine as ME

from lib.test_data_loader import createDataLoader, TestDataset
from lib.trainer_base import load_state, default_restore_location
from lib.contrastive_trainer import NPairLossClusterTrainer

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

def loadEvalFilesFromFile(highLevelFileName):
    with open(highLevelFileName, newline='') as highLevelFile:
        subfiles = highLevelFile.readlines()
        return [subfile.strip() for subfile in subfiles]

def test(testFiles, model, config):
  dataLoader =  createDataLoader(testFiles, config.data.voxel_size, config.data.batch_size)
  model.eval()
  output = []

  lastOutput = 0
  with torch.no_grad():
    for batch in dataLoader:
      coords = batch[0]
      feats = batch[1]
      tensor = ME.SparseTensor(coords=coords, feats=feats)

      outForBatch = model(tensor)
      outFeats = outForBatch.F
      for i in range(outForBatch.shape[0]):
          output.append(outFeats[i, :].numpy())
      numProcessed = len(output)
      if (numProcessed > lastOutput + 1000):
          print("Processed " + str(numProcessed))
          lastOutput = numProcessed

  return output


def outputResults(resultsFileName, testFiles, labels):
    if (len(labels) != len(testFiles)):
        print("Not as many labels generated as test files. ")
        exit(1)
    combinedResults = [(testFiles[i], labels[i]) for i in range(len(testFiles))]
    with open(resultsFileName, 'wb') as resultsFile:
        joblib.dump(combinedResults, resultsFile)


def single_proc_run(config):
  num_feats=1 # LiDAR just has reflectance/intensity
  model = ClusterLabelModel(num_feats, config.net.model_n_out, config, D=3)
  model = model.double()
  # TODO  need to verify that this can fully load model
  # model.updateWithPretrainedWeights(config.net.pretrained_weights)
  state = torch.load(config.net.pretrained_weights, map_location=lambda s, l: default_restore_location(s, 'cpu'))
  load_state(model, state['state_dict'])

  testFiles = loadEvalFilesFromFile(config.data.eval_dataset_file)
  labels = test(testFiles, model, config)
  outputResults(config.data.results_out, testFiles, labels)


if __name__ == "__main__":
  os.environ['MKL_THREADING_LAYER'] = 'GNU'
  main()
