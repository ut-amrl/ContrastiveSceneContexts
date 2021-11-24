import os
import csv
import random
import argparse
import glob

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

def getTrainingSequences():
    return ["00", "01", "02", "03", "04", "05"]

def getAllTrainingFilesForSequence(datasetDir, sequence):
    seqDir = os.path.join(datasetDir, sequence)
    searchPattern = seqDir + "/*points.npy"
    files = glob.glob(searchPattern)
    return files

def getAllTrainingFiles(datasetDir):
    trainingSeqs = getTrainingSequences()
    trainingFiles = []
    for seqNum in trainingSeqs:
        trainingFiles.extend(getAllTrainingFilesForSequence(datasetDir, seqNum))

    return trainingFiles

def extractSemanticClassFromFileName(fileName):
    basename = os.path.basename(fileName)
    semClassComponent = basename.split('_')[-2]
    return int(semClassComponent.replace('semClass', ''))

def generatePairs(numPairsToGenerate, numNegativeEntriesPerPair, datasetDir):

    trainingFiles = getAllTrainingFiles(datasetDir)

    semanticClassGroupList, semanticClassGroupDict = getSemanticClassGroups()

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

def writeSampleGroups(sampleGroups, outFile):
    with open(outFile, 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        for sampleGroup in sampleGroups:
            datawriter.writerow(sampleGroup)


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
    return parser.parse_known_args()

if __name__ == "__main__":
    FLAGS, unparsed = argParser()

    sampleGroups = generatePairs(FLAGS.num_sample_pairs, FLAGS.num_neg_samples_per_entry, FLAGS.dataset_root)
    writeSampleGroups(sampleGroups, FLAGS.output_file)

