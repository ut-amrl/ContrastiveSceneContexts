import csv
import random
import argparse

from semantic_kitti_analysis_utils import getSemanticClassGroups, extractSemanticClassFromFileName, \
    getAllPointCloudFilesForSequences, getCondensedSemanticClassGroups, writeSampleGroups

def getTrainingSequences():
    return ["00", "01", "02", "03", "04", "05"]

def generatePairs(numPairsToGenerate, numNegativeEntriesPerPair, datasetDir, minPoints):

    trainingFiles = getAllPointCloudFilesForSequences(datasetDir, minPoints, getTrainingSequences())

    # semanticClassGroupList, semanticClassGroupDict = getSemanticClassGroups()
    semanticClassGroupList, semanticClassGroupDict = getCondensedSemanticClassGroups()

    trainingFilesByClassGroup = [[] for i in range(len(semanticClassGroupList))]

    sampleGroups = []

    for trainingFile in trainingFiles:
        semanticClassId = extractSemanticClassFromFileName(trainingFile)
        classGroupIndex = semanticClassGroupDict[semanticClassId]
        trainingFilesByClassGroup[classGroupIndex].append(trainingFile)

    numFilesPerClassGroup = [len(classGroupFiles) for classGroupFiles in trainingFilesByClassGroup]
    if (not list(filter((0).__ne__, numFilesPerClassGroup))):
        print("Only one type of class. Exiting")
        exit(1)

    for i in range(numPairsToGenerate):
        sampleIFiles = []
        posSampleA = random.choice(trainingFiles)
        posSampleAClass = extractSemanticClassFromFileName(posSampleA)
        semanticClassGroupIndex = semanticClassGroupDict[posSampleAClass]
        posSampleB = random.choice(trainingFilesByClassGroup[semanticClassGroupIndex])

        sampleIFiles.append(posSampleA)
        sampleIFiles.append(posSampleB)

        otherClassIndices = list(range(len(semanticClassGroupList)))
        otherClassIndices.remove(semanticClassGroupIndex)

        for j in range(numNegativeEntriesPerPair):
            negSemanticClassGroupIndex = random.choice(otherClassIndices)
            filesForClassGroup = trainingFilesByClassGroup[negSemanticClassGroupIndex]
            while (not filesForClassGroup):
                negSemanticClassGroupIndex = random.choice(otherClassIndices)
                filesForClassGroup = trainingFilesByClassGroup[negSemanticClassGroupIndex]

            sampleIFiles.append(random.choice(filesForClassGroup))

        sampleGroups.append(sampleIFiles)

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
        default=0,
        help='Minimum points in an instance for the file to be used',
    )
    return parser.parse_known_args()

if __name__ == "__main__":
    FLAGS, unparsed = argParser()

    sampleGroups = generatePairs(FLAGS.num_sample_pairs, FLAGS.num_neg_samples_per_entry, FLAGS.dataset_root, FLAGS.min_points)
    writeSampleGroups(sampleGroups, FLAGS.output_file)

