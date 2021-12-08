
from model.cluster_label_model import ClusterLabelModel
import torch
from omegaconf import OmegaConf
import hydra
import os
import logging
import joblib
import csv

import lib.unoriginal.multiprocessing_utils as mpu
import numpy as np
import MinkowskiEngine as ME

from lib.classification_data_loader import createClassificationDataLoader

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

def test(fileListFileName, model, config):
  dataLoader =  createClassificationDataLoader(fileListFileName, config.data.voxel_size, config.data.batch_size,
                                               config.misc.num_gpus, includeLabels=False, train=False)
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

def loadEvalFilesFromFile(highLevelFileName):
    pointCloudFiles = []
    with open(highLevelFileName, newline='') as samplesFile:
        samplesReader = csv.reader(samplesFile)
        for row in samplesReader:
            trimmedRow = [entry.strip() for entry in row]
            pointCloudFileName = trimmedRow[0]
            pointCloudFiles.append(pointCloudFileName)
    return pointCloudFiles


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
  model.updateWithPretrainedWeights(config.net.pretrained_weights)

  labels = test(config.data.eval_dataset_file, model, config)
  testFiles = loadEvalFilesFromFile(config.data.eval_dataset_file)
  outputResults(config.data.results_out, testFiles, labels)


if __name__ == "__main__":
  os.environ['MKL_THREADING_LAYER'] = 'GNU'
  main()
