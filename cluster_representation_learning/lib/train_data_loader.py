
from torch.utils.data import Dataset, DataLoader
import numpy as np
import MinkowskiEngine as ME

class TrainDataset(Dataset):

    def __init__(self, datafiles, voxelSize):
        # Voxel size much match training voxel size
        self.datafiles = datafiles
        self.voxelSize = voxelSize

    def loadFromFile(self, fileName):
        # TODO
        return []

    def __getitem__(self, i):
        # TODO do we want to load things into file or cache them or just read from file as needed
        datafile = self.datafiles[i]

        return self.loadFromFile(datafile)

    def __len__(self):
        # TODO need to manage transformations and comparing multiple
        return len(self.datafiles)

def createDataLoader(datafiles, voxelSize, batchSize):
    testDataSet = TrainDataset(datafiles, voxelSize)

    return DataLoader(testDataSet, batchSize, shuffle=True, collate_fn=ME.utils.batch_sparse_collate)