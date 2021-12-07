import argparse
import csv

from semantic_kitti_analysis_utils import getAllPointCloudFilesForSequences, getCondensedSemanticClassGroups, extractSemanticClassFromFileName

def getTrainingSequences():
    return ["00", "01", "02", "03", "04", "05"]
    # return ["02"]

def getTuplesOfFileNameAndClassIndex(pointCloudFiles):

    semanticClassGroupList, semanticClassGroupDict = getCondensedSemanticClassGroups()

    return [[pointCloudFile, semanticClassGroupDict[extractSemanticClassFromFileName(pointCloudFile)]] for pointCloudFile in pointCloudFiles]

def writeClassificationTrainingDataToFile(classificationTrainingTuples, outputFile):
    with open(outputFile, 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        for classificationTrainingTuple in classificationTrainingTuples:
            datawriter.writerow(classificationTrainingTuple)

def argParser():
    parser = argparse.ArgumentParser("./semantic_kitti_generate_classification_results_list.py")
    parser.add_argument(
        '--dataset_root', '-d',
        type=str,
        required=True,
        help='Directory to dataset root',
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

    pointCloudFiles = getAllPointCloudFilesForSequences(FLAGS.dataset_root, FLAGS.min_points, getTrainingSequences())
    classificationTrainingTuples = getTuplesOfFileNameAndClassIndex(pointCloudFiles)
    writeClassificationTrainingDataToFile(classificationTrainingTuples, FLAGS.output_file)

