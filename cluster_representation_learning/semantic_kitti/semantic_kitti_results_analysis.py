import argparse
import random
import numpy as np

from numpy import linalg as LA

import matplotlib.pyplot as plt

import joblib

from semantic_kitti_analysis_utils import getSemanticClassGroups, extractSemanticClassFromFileName

def argParser():
    parser = argparse.ArgumentParser("./semantic_kitti_results_analysis.py")
    parser.add_argument(
        '--learned_features', '-l',
        type=str,
        required=True,
        help='File name containing point cloud files and the representation for the point cloud',
    )

    parser.add_argument(
        '--target_class', '-t',
        type=int,
        required=True,
        help='Numeric id from semantic kitti of the class that we should be computing results for',
    )

    parser.add_argument(
        '--num_matching_samples', '-m',
        type=int,
        required=True,
        help='Number of samples from the target class that should have distance computed'
    )
    parser.add_argument(
        '--num_negative_samples', '-n',
        type=int,
        required=True,
        help='Number of samples from classes other than the target that should have distance computed'
    )

    return parser.parse_known_args()

def readResults(resultsFileName):
    return joblib.load(resultsFileName)

def getSamplesForAndNotForClass(resultsWithFileNames, targetClassGroupIndex, semanticClassDict):
    samplesForSameClassGroup = []
    samplesForOtherClassGroup = []
    for pointCloudFileName, feature in resultsWithFileNames:
        semanticClassForFile = extractSemanticClassFromFileName(pointCloudFileName)
        classGroupIndexForFile = semanticClassDict[semanticClassForFile]
        if (classGroupIndexForFile == targetClassGroupIndex):
            samplesForSameClassGroup.append((pointCloudFileName, feature))
        else:
            samplesForOtherClassGroup.append((pointCloudFileName, feature))
    return samplesForSameClassGroup, samplesForOtherClassGroup

def computeSampleDistance(feature1, feature2):
    return LA.norm(feature1 - feature2)

def getIntraClassDistances(samplesForSameClassGroup, numIntraClassComparisons):
    numSamples = len(samplesForSameClassGroup)
    maximumPossibleComparisons = (numSamples * (numSamples - 1)) / 2
    featureDists = []
    if (numIntraClassComparisons > maximumPossibleComparisons):
        for i in range(numSamples):
            for j in range(numSamples):
                if (i != j):
                    _, feature1 = samplesForSameClassGroup[i]
                    _, feature2 = samplesForSameClassGroup[j]
                    featureDists.append(computeSampleDistance(feature1, feature2))
    else:
        # TODO consider keeping track of list of pairs so we don't duplicate (random sampling without replacement instead)
        for i in range(numIntraClassComparisons):
            sampleIndex1 = random.randint(0, numSamples - 1)
            sampleIndex2 = random.randint(0, numSamples - 1)
            while (sampleIndex2 == sampleIndex1):
                sampleIndex2 = random.randint(0, numSamples - 1)
            _, feature1 = samplesForSameClassGroup[sampleIndex1]
            _, feature2 = samplesForSameClassGroup[sampleIndex2]
            featureDists.append(computeSampleDistance(feature1, feature2))
    return featureDists

def getInterClassDistances(samplesForSameClassGroup, samplesForOtherClassGroups, numInterClassComparisons):
    maximumPossibleComparisons = len(samplesForSameClassGroup) * len(samplesForOtherClassGroups)

    featureDists = []
    if (numInterClassComparisons > maximumPossibleComparisons):
        for targetClassPointCloudFile, targetClassFeature in samplesForSameClassGroup:
            for otherClassPointCloudFile, otherClassFeature in samplesForSameClassGroup:
                featureDists.append(computeSampleDistance(targetClassFeature, otherClassFeature))
    else:
        for i in range(numInterClassComparisons):
            _, targetClassFeature = random.choice(samplesForSameClassGroup)
            _, otherClassFeature = random.choice(samplesForOtherClassGroups)
            featureDists.append(computeSampleDistance(targetClassFeature, otherClassFeature))

    return featureDists

def getCDFData(dataset, num_bins):

    # getting data of the histogram
    count, bins_count = np.histogram(dataset, bins=num_bins)

    # finding the PDF of the histogram using count values

    pdf = count / sum(count)


    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    cdf = np.insert(cdf, 0, 0)

    max_val = np.amax(dataset)

    return (cdf, bins_count, max_val)

def plotCDFs(intraClassDistances, interClassDistances, bins=80):

    interClassCDFData, bins_count, interClassMax = getCDFData(interClassDistances, bins)
    plt.plot(bins_count, interClassCDFData, linestyle='dashed', label="Distances from other classes")

    intraClassDistancesCDFData, bins_count, intraClassMax = getCDFData(intraClassDistances, bins)
    plt.plot(bins_count, intraClassDistancesCDFData, linestyle='solid', label="Distances between same class features")

    x_lim = intraClassMax
    plt.xlim(0, x_lim)

    plt.legend()
    plt.ylim(0, 1)
    plt.title("Distances between features of the same class and different classes")
    plt.xlabel("Feature distance")
    plt.ylabel("Proportion of data")

if __name__ == "__main__":
    FLAGS, unparsed = argParser()

    resultsFile = FLAGS.learned_features
    targetClass = FLAGS.target_class
    numMatchingSamples = FLAGS.num_matching_samples
    numNegativeSamples = FLAGS.num_negative_samples


    semanticClassList, semanticClassDict = getSemanticClassGroups()

    targetClassGroupIndex = semanticClassDict[targetClass]

    resultsWithFileNames = readResults(resultsFile)

    samplesForSameClassGroup, samplesForOtherClassGroup = getSamplesForAndNotForClass(resultsWithFileNames, targetClassGroupIndex, semanticClassDict)

    intraClassDistances = getIntraClassDistances(samplesForSameClassGroup, numMatchingSamples)
    interClassDistances = getInterClassDistances(samplesForSameClassGroup, samplesForOtherClassGroup, numNegativeSamples)

    plotCDFs(intraClassDistances, interClassDistances)






