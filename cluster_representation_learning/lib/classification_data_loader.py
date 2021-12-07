import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import MinkowskiEngine as ME
from lib.data_loading_utils import loadPointCloudFromFile
from lib.unoriginal.data_sampler import DistributedInfSampler

import csv

class ClassificationDataset(Dataset):

    def __init__(self, samplesFileName, voxelSize, batchSize):
        # Voxel size much match training voxel size
        self.voxelSize = voxelSize
        self.batchSize = batchSize
        self.loadAllData(samplesFileName)

    def loadAllData(self, samplesFileName):
        self.pointCloudCoords = []
        self.pointCloudFeats = []
        self.labels = []

        with open(samplesFileName, newline='') as samplesFile:
            samplesReader = csv.reader(samplesFile)
            for row in samplesReader:
                if (len(row) != 2):
                    print("Each entry needs a point cloud file and the label for the point cloud (should be in the range [0, C-1] for C classes, but entry was " )
                    print(row)
                    continue

                trimmedRow = [entry.strip() for entry in row]
                pointCloudFileName = trimmedRow[0]
                label = int(trimmedRow[1])

                coords, feats = loadPointCloudFromFile(pointCloudFileName, self.voxelSize)
                self.pointCloudCoords.append(coords)
                self.pointCloudFeats.append(feats)
                self.labels.append(label)

        self.labels = np.stack(self.labels, axis=0)
        self.labels = np.reshape(self.labels, (len(self.pointCloudCoords), 1))

    def __getitem__(self, i):
        return (self.pointCloudCoords[i], self.pointCloudFeats[i], self.labels[i])

    def __len__(self):
        return len(self.labels)

def createClassificationDataLoader(pointCloudFilesAndLabelFileName, voxelSize, batchSize, numGpus):
    # print("Creating training data set")
    trainingDataSet = ClassificationDataset(pointCloudFilesAndLabelFileName, voxelSize, batchSize)
    if (numGpus > 1):
        sampler = DistributedInfSampler(trainingDataSet)
    else:
        sampler = None
    # print("Creating training data loader")
    shuffle = False if sampler else True
    return DataLoader(trainingDataSet, batchSize, shuffle=shuffle, collate_fn=ME.utils.batch_sparse_collate, sampler=sampler)
