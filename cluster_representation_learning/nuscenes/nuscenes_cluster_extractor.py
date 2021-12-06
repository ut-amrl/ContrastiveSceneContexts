import argparse

from nuscenes import NuScenes

def argParser():
    parser = argparse.ArgumentParser("./nuscenes_cluster_extractor.py")

    parser.add_argument(
        '--dataset_root', '-d',
        type=str,
        required=True,
        help='Directory that is the root of the nuscenes dataset',
    )

    parser.add_argument(
        '--subset', '-s',
        type=str,
        required=False,
        default='v1.0-trainval',
        help='Options: v1.0-trainval, v1.0-test, v1.0-mini',
    )

    return parser.parse_known_args()


def main(argParserResults):

    nusc = NuScenes(version=argParserResults.subset, dataroot=argParserResults.dataset_root, verbose=True)

if __name__ == "__main__":
    FLAGS, unparsed = argParser()

    main(FLAGS)
