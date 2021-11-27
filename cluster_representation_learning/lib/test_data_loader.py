
from torch.utils.data import Dataset, DataLoader
import numpy as np
import MinkowskiEngine as ME
from lib.data_loading_utils import loadPointCloudFromFile

class TestDataset(Dataset):

    def __init__(self, datafiles, voxelSize):
        # Voxel size much match training voxel size
        self.datafiles = datafiles
        self.voxelSize = voxelSize

    def __getitem__(self, i):
        # TODO do we want to read these all in on start up, or cache as we read, or just read them from file as needed
        datafile = self.datafiles[i]

        return loadPointCloudFromFile(datafile, self.voxelSize)

    def __len__(self):
        return len(self.datafiles)

def createDataLoader(datafiles, voxelSize, batchSize):
    testDataSet = TestDataset(datafiles, voxelSize)

    return DataLoader(testDataSet, batchSize, shuffle=False, collate_fn=ME.utils.batch_sparse_collate)