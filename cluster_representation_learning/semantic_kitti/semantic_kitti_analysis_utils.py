import os
import glob
import numpy as np
import csv

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


def getCondensedSemanticClassGroups():
    semanticClassList = [
        [10, 252, 18, 258, 20, 259], # Car/Truck/Other vehicle
        [11, 15, 31, 253, 32, 255], # Bike/Motorcycle/Bicyclist/Motorcyclist
        [13, 257], # Bus
        [16, 256], # On rails
        [30, 254], # Person
    ]

    semanticClassDict = {}
    for i in range(len(semanticClassList)):
        classGroup = semanticClassList[i]
        for classIndex in classGroup:
            semanticClassDict[classIndex] = i

    return semanticClassList, semanticClassDict

def getLabelsForClassIndices():
    semanticClassLabelsList = [
        "Car",
        "Bike",
        "Bus",
        "Motorcycle",
        "On rails",
        "Truck",
        "Other Vehicle",
        "Person",
        "Bicyclist",
        "Motorcyclist"
    ]
    return semanticClassLabelsList

def getLabelsForCondensedClassIndices():
    semanticClassLabelsList = [
        "Car/Truck/OtherVehicle",
        "Bike/Motorcycle/Bicyclist/Motorcyclist",
        "Bus",
        "On rails",
        "Person"
    ]
    return semanticClassLabelsList

def getLabelForClassNum(classNum):
    _, semanticClassDict = getSemanticClassGroups()
    return getLabelsForClassIndices()[semanticClassDict[classNum]]

def getLabelForCondensedClassNum(classNum):
    _, semanticClassDict = getCondensedSemanticClassGroups()
    return getLabelsForCondensedClassIndices()[semanticClassDict[classNum]]

def extractSemanticClassFromFileName(fileName):
    basename = os.path.basename(fileName)
    semClassComponent = basename.split('_')[-2]
    return int(semClassComponent.replace('semClass', ''))

def extractInstanceFromFileName(fileName):
    basename = os.path.basename(fileName)
    instanceComponent = basename.split('_')[-3]
    return int(instanceComponent.replace('inst', ''))

def writeSampleGroups(sampleGroups, outFile):
    with open(outFile, 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        for sampleGroup in sampleGroups:
            datawriter.writerow(sampleGroup)

def getFilesOverMinPoints(filesList, minPoints):
    if (minPoints == 0):
        return filesList, 0

    keepFiles = []
    numExcluded = 0
    for fileName in filesList:
        coordsWithFeats = np.load(fileName)
        if (coordsWithFeats.shape[0] < minPoints):
            numExcluded += 1
        else:
            keepFiles.append(fileName)
    return keepFiles, numExcluded

def getAllPointCloudFilesForSequence(datasetDir, sequence, minPoints):
    seqDir = os.path.join(datasetDir, sequence)
    searchPattern = seqDir + "/*points.npy"
    files = glob.glob(searchPattern)

    keepFiles, numExcluded = getFilesOverMinPoints(files, minPoints)

    print("Sequence " + sequence + ": Excluded " + str(numExcluded) + " files because they had less than " + str(minPoints))
    print("Kept " + str(len(keepFiles)) + " files")
    return keepFiles

def getAllPointCloudFilesForSequences(datasetDir, minPoints, sequences):
    pointCloudFiles = []
    for seqNum in sequences:
        pointCloudFiles.extend(getAllPointCloudFilesForSequence(datasetDir, seqNum, minPoints))

    return pointCloudFiles