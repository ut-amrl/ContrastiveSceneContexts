import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import MinkowskiEngine as ME
from lib.data_loading_utils import loadPointCloudFromFile
from lib.unoriginal.data_sampler import DistributedInfSampler

import csv

class TrainDataset(Dataset):

    def __init__(self, matchesFileName, voxelSize, batchSize):
        # Voxel size much match training voxel size
        self.voxelSize = voxelSize
        self.batchSize = batchSize
        self.loadAllData(matchesFileName)

    def loadAllData(self, matchesFileName):
        self.pointCloudCoords = []
        self.pointCloudFeats = []
        self.sampleGroups = []
        self.fileNames = {}
        self.fileNameList = []

        with open(matchesFileName, newline='') as matchesFile:
            matchesReader = csv.reader(matchesFile)
            for row in matchesReader:

                if (len(row) < 3):
                    print("Each entry needs at least 2 postive samples and 1 negative, but entry was only: ")
                    print(row)
                    continue

                trimmedRow = [pointCloudFileName.strip() for pointCloudFileName in row]
                for pointCloudFileName in trimmedRow:
                    if (pointCloudFileName not in self.fileNames.keys()):
                        self.fileNames[pointCloudFileName] = len(self.pointCloudCoords)
                        # TODO add error handling for if file is invalid (skip this entry)
                        coords, feats = loadPointCloudFromFile(pointCloudFileName, self.voxelSize)
                        self.pointCloudCoords.append(coords)
                        self.pointCloudFeats.append(feats)
                        self.fileNameList.append(pointCloudFileName)

                sampleGroupUsingIndices = [self.fileNames[pointCloudFileName] for pointCloudFileName in trimmedRow]
                self.sampleGroups.append(sampleGroupUsingIndices)

    def __getitem__(self, i):
        # TODO do we want to load things into file or cache them or just read from file as needed
        sampleGroup = self.sampleGroups[i]
        # print("Sample group!")
        # print(sampleGroup)
        posCloudCoordsA = self.pointCloudCoords[sampleGroup[0]]
        posCloudFeatsA = self.pointCloudFeats[sampleGroup[0]]

        posCloudCoordsB = self.pointCloudCoords[sampleGroup[1]]
        posCloudFeatsB = self.pointCloudFeats[sampleGroup[1]]

        negSamples = []
        for i in range(2, len(sampleGroup)):
            negSamples.append([self.pointCloudCoords[i], self.pointCloudFeats[i]])

        #print("Relevant files " + str([self.fileNameList[i] for i in sampleGroup]))
        return (posCloudCoordsA, posCloudFeatsA, posCloudCoordsB, posCloudFeatsB, negSamples, [self.fileNameList[i] for i in sampleGroup])

    def __len__(self):
        # TODO need to manage transformations and comparing multiple
        return len(self.sampleGroups)

def collateTrainingData(training_data_entries):
    # ME.utils.batch_sparse_collate
    # TODO

    # posCloudCoordsAs - List with each entry as the coordiates for part A of the positive pair
    # posCloudCoordsBs - List with each entry as the coordiates for part B of the positive pair
    # posCloudFeatsAs - List with each entry having the features for part A of the positive pair
    # posCloudFeatsBs - List with each entry having the features for part B of the positive pair
    # negSamples - List with each entry as a tuple of (coord, feature) for the negative entries
    posCloudCoordsAs, posCloudFeatsAs, posCloudCoordsBs, posCloudFeatsBs, negSamplesList, fileNamesList = list(zip(*training_data_entries))

    # print("Batch size!")
    # print(len(posCloudCoordsAs))
    # print([len(sublist) for sublist in negSamplesList])

    feats = []
    coords = []
    posCoordsList = []
    negSamplesBatchIdsList = []
    fileNamesBatchList = []

    nextBatchNum = 0

    for batchId, posCloudCoordA in enumerate(posCloudCoordsAs):
        fileNamesBatchList.extend(fileNamesList[batchId])
        N0 = posCloudCoordsAs[batchId].shape[0]
        N1 = posCloudCoordsBs[batchId].shape[0]
        # print("N0 for batch " + str(batchId) + ": " + str(N0))
        # print("N1 for batch " + str(batchId) + ": " + str(N1))

        # For every set (2 positive and M negative) we need to have one entry that gives the index of the positive coordinates
        posCoordsList.append(nextBatchNum + 1)
        posCoordsList.append(-1)

        # Add the features to the list of features
        feats.append(torch.from_numpy(posCloudFeatsAs[batchId]))
        feats.append(torch.from_numpy(posCloudFeatsBs[batchId]))

        # Prepend the coordinates with a batch id
        # This will be different than the processing batch -- each point cloud we process gets its own index
        # This is to ensure that we generate N feature vectors for the N different point clouds (comprised of positive
        # and negative that we need to process)
        batchedCoordsA = torch.cat((torch.ones(N0, 1).float() * nextBatchNum,
                                    torch.from_numpy(posCloudCoordA).float()), 1)
        nextBatchNum += 1
        batchedCoordsB = torch.cat((torch.ones(N1, 1).float() * nextBatchNum,
                   torch.from_numpy(posCloudCoordsBs[batchId]).float()), 1)
        nextBatchNum += 1

        coords.append(batchedCoordsA)
        coords.append(batchedCoordsB)

        negSamplesForBatch = negSamplesList[batchId]
        negSamplesBatchIds = []
        for negSampleCoords, negSampleFeats in negSamplesForBatch:
            feats.append(torch.from_numpy(negSampleFeats))
            negSamplesBatchIds.append(nextBatchNum)

            pointCloudSize = negSampleCoords.shape[0]
            # print("Size for batch " + str(batchId) + ": " + str(pointCloudSize))

            # Prepend the coordinates with a batch id
            # This will be different than the processing batch -- each point cloud we process gets its own index
            # This is to ensure that we generate N feature vectors for the N different point clouds (comprised of positive
            # and negative that we need to process)
            batchedNegCoords = torch.cat((torch.ones(pointCloudSize, 1).float() * nextBatchNum,
                                        torch.from_numpy(negSampleCoords).float()), 1)
            nextBatchNum += 1
            coords.append(batchedNegCoords)
            posCoordsList.append(-1)

        negSamplesBatchIdsList.append(negSamplesBatchIds)
        negSamplesBatchIdsList.append([]) # Don't have any negative samples for the second positive value
        for i in range(len(negSamplesForBatch)):
            negSamplesBatchIdsList.append([]) # Don't have any negative samples for any of the negative values

    feats = torch.cat(feats, 0).double()
    coords = torch.cat(coords, 0).double()
    # print("Coords")
    # print(coords)
    # print("Pos coords list!")
    # print(posCoordsList)
    posCoordsList = torch.Tensor(posCoordsList).long()
    # print("Neg list")
    # print(negSamplesBatchIdsList)

    return {
        'feats':feats,
        'coords':coords,
        'posMatches':posCoordsList,
        'negMatches':negSamplesBatchIdsList,
        'fileNames':fileNamesBatchList
    }

def createTrainingDataLoader(matchesFileName, voxelSize, batchSize, numGpus):
    # print("Creating training data set")
    trainingDataSet = TrainDataset(matchesFileName, voxelSize, batchSize)
    if (numGpus > 1):
        sampler = DistributedInfSampler(trainingDataSet)
    else:
        sampler = None
    # print("Creating training data loader")
    shuffle = False if sampler else True
    return DataLoader(trainingDataSet, batchSize, shuffle=shuffle, collate_fn=collateTrainingData, sampler=sampler)
