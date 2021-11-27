import argparse

from semantic_kitti_analysis_utils import getAllPointCloudFilesForSequences

def getEvalSequences():
    return ["06", "07", "08", "09", "10"]

def writeEvalFiles(evalFiles, outFileName):
    with open(outFileName, 'w', newline='') as outFile:
        outFile.write('\n'.join(evalFiles))

def argParser():
    parser = argparse.ArgumentParser("./semantic_kitti_generate_eval_files_list.py")
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

    evalFiles = getAllPointCloudFilesForSequences(FLAGS.dataset_root, FLAGS.min_points, getEvalSequences())
    writeEvalFiles(evalFiles, FLAGS.output_file)

