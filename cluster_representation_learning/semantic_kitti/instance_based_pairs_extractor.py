import csv
import random
import argparse
import numpy as np
import open3d

from semantic_kitti_analysis_utils import extractInstanceFromFileName, writeSampleGroups, getAllPointCloudFilesForSequences

def getTrainingSequences():
    return ["00", "01", "02", "03", "04", "05"]

def checkClusterDimension(posSampleAPoints, posSampleBPoints, candidateNegSamplePoints, minSizeRatio):
    posSampleACoords = open3d.utility.Vector3dVector(posSampleAPoints[:, 0:3])
    posSampleBCoords = open3d.utility.Vector3dVector(posSampleBPoints[:, 0:3])
    negSampleCoords = open3d.utility.Vector3dVector(candidateNegSamplePoints[:, 0:3])

    print(posSampleAPoints.shape[0])
    print(posSampleBPoints.shape[0])
    print(candidateNegSamplePoints.shape[0])

    bbA = open3d.geometry.OrientedBoundingBox.create_from_points(posSampleACoords)
    bbB = open3d.geometry.OrientedBoundingBox.create_from_points(posSampleBCoords)
    negSampleBB = open3d.geometry.OrientedBoundingBox.create_from_points(negSampleCoords)

    ratioNegWithA = bbA.extent[0] / negSampleBB.extent[0]
    if (ratioNegWithA < 1):
        ratioNegWithA = 1 / ratioNegWithA
    ratioNegWithB = bbB.extent[0] / negSampleBB.extent[0]
    if (ratioNegWithB < 1):
        ratioNegWithB = 1 / ratioNegWithB

    if (ratioNegWithA > minSizeRatio) and (ratioNegWithB > minSizeRatio):
        return True
    return False

def generatePairs(numPairsToGenerate, numNegativeEntriesPerPair, datasetDir, minPoints, minSizeRatio):

    trainingFiles = getAllPointCloudFilesForSequences(datasetDir, minPoints, getTrainingSequences())

    filesByInstance = {}

    for trainingFile in trainingFiles:
        instanceId = extractInstanceFromFileName(trainingFile)
        filesForInstance = []
        if (instanceId in filesByInstance):
            filesForInstance = filesByInstance[instanceId]
        filesForInstance.append(trainingFile)
        filesByInstance[instanceId] = filesForInstance

    if (len(filesByInstance) <= 1):
        print("Only one instance. Exiting")
        exit(1)

    sampleGroups = []

    for i in range(numPairsToGenerate):
        samplesI = []
        posSampleA = random.choice(trainingFiles)

        instanceForSample = extractInstanceFromFileName(posSampleA)
        while (len(filesByInstance[instanceForSample]) <= 1):
            posSampleA = random.choice(trainingFiles)
            instanceForSample = extractInstanceFromFileName(posSampleA)

        filesForInstance = filesByInstance[instanceForSample]
        posSampleB = random.choice(filesForInstance)

        posSampleAPoints = np.load(posSampleA)

        while (posSampleA == posSampleB):
            posSampleB = random.choice(filesForInstance)

        posSampleBPoints = np.load(posSampleB)

        samplesI.append(posSampleA)
        samplesI.append(posSampleB)
        for i in range(numNegativeEntriesPerPair):
            foundNegSample = False
            negSample = None
            while (not foundNegSample):
                negSample = random.choice(trainingFiles)
                if (extractInstanceFromFileName(negSample) == instanceForSample):
                    continue
                candidateNegSamplePoints = np.load(negSample)
                if (checkClusterDimension(posSampleAPoints, posSampleBPoints, candidateNegSamplePoints, minSizeRatio)):
                    foundNegSample = True
            samplesI.append(negSample)

        sampleGroups.append(samplesI)
    return sampleGroups

def argParser():
    parser = argparse.ArgumentParser("./semantic_class_based_pairs_extractor.py")
    parser.add_argument(
        '--dataset_root', '-d',
        type=str,
        required=True,
        help='Directory to dataset root',
    )
    parser.add_argument(
        '--num_neg_samples_per_entry', '-n',
        type=int,
        required=True,
        default="config/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--num_sample_pairs', '-s',
        type=int,
        required=True,
        default="config/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--output_file', '-o',
        dest='output_file',
        type=str,
        required=True,
        default="./output_file.csv",
        help='Output %(default)s',
    )
    parser.add_argument(
        '--min_points_in_inst', '-m',
        dest='min_points',
        type=int,
        required=False,
        default=4,
        help='Minimum points in an instance for the file to be used',
    )

    parser.add_argument(
        '--bounding_box_ratio', '-r',
        type=int,
        required=False,
        default=1.5,
        help='Minimum points in an instance for the file to be used',
    )

    return parser.parse_known_args()

if __name__ == "__main__":
    FLAGS, unparsed = argParser()

    sampleGroups = generatePairs(FLAGS.num_sample_pairs, FLAGS.num_neg_samples_per_entry, FLAGS.dataset_root,
                                 FLAGS.min_points, FLAGS.bounding_box_ratio)
    writeSampleGroups(sampleGroups, FLAGS.output_file)

