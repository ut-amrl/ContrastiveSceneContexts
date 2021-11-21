
from torch.utils.data import Dataset, DataLoader
import numpy as np
import MinkowskiEngine as ME

class TestDataset(Dataset):

    def __init__(self, datafiles, voxelSize):
        # Voxel size much match training voxel size
        self.datafiles = datafiles
        self.voxelSize = voxelSize

    def loadFromFile(self, fileName):
        coordsWithFeats = np.load(fileName)
        coords = coordsWithFeats[:, 0:3]
        feats = np.reshape(coordsWithFeats[:, 3], (coordsWithFeats.shape[0], 1))

        # Quantize the input (TODO check that this is combining features appropriately when multiple points in same voxel)
        discrete_coords, unique_feats = ME.utils.sparse_quantize(
            coords=coords,
            feats=feats,
            quantization_size=self.voxelSize)
        return discrete_coords, unique_feats

    def __getitem__(self, i):
        # TODO do we want to read these all in on start up, or cache as we read, or just read them from file as needed
        datafile = self.datafiles[i]

        return self.loadFromFile(datafile)

    def __len__(self):
        return len(self.datafiles)

def createDataLoader(datafiles, voxelSize, batchSize):
    testDataSet = TestDataset(datafiles, voxelSize)

    return DataLoader(testDataSet, batchSize, collate_fn=ME.utils.batch_sparse_collate)