import csv
import argparse

import numpy as np
import torch

import joblib

def readResults(resultsFileName):
    return joblib.load(resultsFileName)

def argParser():
    parser = argparse.ArgumentParser("./output_classification_accuracy.py")
    parser.add_argument(
        '--classification_results', '-c',
        type=str,
        required=True,
        help='File with classifications results',
    )

    parser.add_argument(
        '--classification_ground_truth', '-t',
        type=str,
        required=True,
        help='File with classification ground truth',
    )

    return parser.parse_known_args()

def getClassificationResultsFromOneHotEncoding(classificationResultsMatrix):
    return torch.max(classificationResultsMatrix)[1]

def buildResultMatrix(resultsAsTuples):
    concatList = [classificationVal for fileName, classificationVal in resultsAsTuples]
    return np.concatenate(concatList)

def getClassificationResultsByFilename(classificationResultsFile):
    classificationResults = {}
    rawResultsAsTuples = readResults(classificationResultsFile)
    resultMatrix = np.concatenate(rawResultsAsTuples, axis=0)
    labelsFromResultMat = getClassificationResultsFromOneHotEncoding(resultMatrix)

    for i in range(len(rawResultsAsTuples)):
        fileName = rawResultsAsTuples[i][0]
        classificationResults[fileName] = labelsFromResultMat[i]

    return classificationResults

def getGroundTruthClassificationByFile(groundTruthClassificationFile):
    groundTruthLabelsByFile = {}

    with open(groundTruthClassificationFile, newline='') as samplesFile:
        samplesReader = csv.reader(samplesFile)
        for row in samplesReader:
            if (len(row) != 2):
                print(
                    "Each entry needs a point cloud file and the label for the point cloud (should be in the range [0, C-1] for C classes, but entry was ")
                print(row)
                continue

            trimmedRow = [entry.strip() for entry in row]
            pointCloudFileName = trimmedRow[0]
            label = int(trimmedRow[1])

            groundTruthLabelsByFile[pointCloudFileName] = label

    return groundTruthLabelsByFile

def computeAccuracy(classificationResults, classificationGroundTruth):
    numCorrectLabels = 0
    numSamples = 0
    for classificationFile, estimatedLabel in classificationResults:
        if (classificationFile in classificationGroundTruth):
            numSamples += 1
            if (estimatedLabel == classificationGroundTruth[classificationFile]):
                numCorrectLabels += 1
        else:
            print("No ground truth for file " + classificationFile)
    accuracy = numCorrectLabels / numSamples
    print("Accuracy: " + str(accuracy))

if __name__ == "__main__":
    arguments, _ = argParser()

    classificationResults = getClassificationResultsByFilename(arguments.classification_results)
    classificationGroundTruth = getGroundTruthClassificationByFile(arguments.classification_ground_truth)

    computeAccuracy(classificationResults, classificationGroundTruth)



