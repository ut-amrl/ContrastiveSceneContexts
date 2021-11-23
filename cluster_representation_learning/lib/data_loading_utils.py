import numpy as np
import MinkowskiEngine as ME

def loadPointCloudFromFile(fileName, voxelSize):
    coordsWithFeats = np.load(fileName)
    coords = coordsWithFeats[:, 0:3]
    feats = np.reshape(coordsWithFeats[:, 3], (coordsWithFeats.shape[0], 1))

    # Quantize the input (TODO check that this is combining features appropriately when multiple points in same voxel)
    discrete_coords, unique_feats = ME.utils.sparse_quantize(
        coords=coords,
        feats=feats,
        quantization_size=voxelSize)
    return discrete_coords, unique_feats