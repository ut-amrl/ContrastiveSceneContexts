import os
import glob
import numpy as np

def getSemanticClassGroups():
    semanticClassList = [
        [10, 252], # Car
        [11], # Bike
        [13, 257], # Bus
        [15], # Motorcycle
        [16, 256], # On rails
        [18, 258], # Truck
        [20, 259], # Other vehicle
        [30, 254], # Person
        [31, 253], # Bicyclist
        [32, 255] # Motorcyclist
    ]

    semanticClassDict = {}
    for i in range(len(semanticClassList)):
        classGroup = semanticClassList[i]
        for classIndex in classGroup:
            semanticClassDict[classIndex] = i

    return semanticClassList, semanticClassDict

def extractSemanticClassFromFileName(fileName):
    basename = os.path.basename(fileName)
    semClassComponent = basename.split('_')[-2]
    return int(semClassComponent.replace('semClass', ''))

def getAllPointCloudFilesForSequence(datasetDir, sequence, minPoints):
    seqDir = os.path.join(datasetDir, sequence)
    searchPattern = seqDir + "/*points.npy"
    files = glob.glob(searchPattern)

    if (minPoints > 0):
        keepFiles = []
        numExcludedForSeq = 0
        for fileName in files:
            coordsWithFeats = np.load(fileName)
            if (coordsWithFeats.shape[0] < minPoints):
                numExcludedForSeq += 1
            else:
                keepFiles.append(fileName)
        print("Sequence " + sequence + ": Excluded " + str(numExcludedForSeq) + " files because they had less than " + str(minPoints))
        print("Kept " + str(len(keepFiles)) + " files")
        return keepFiles
    else:
        return files

def getAllPointCloudFilesForSequences(datasetDir, minPoints, sequences):
    pointCloudFiles = []
    for seqNum in sequences:
        pointCloudFiles.extend(getAllPointCloudFilesForSequence(datasetDir, seqNum, minPoints))

    return pointCloudFiles