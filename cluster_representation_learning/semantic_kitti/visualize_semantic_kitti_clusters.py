import numpy as np
import argparse
import open3d
from semantic_kitti_analysis_utils import getSemanticClassGroups, extractSemanticClassFromFileName, getFilesOverMinPoints, getLabelForClassNum, getCondensedSemanticClassGroups, getLabelForCondensedClassNum
from multiprocessing import Process

def loadEvalFilesFromFile(highLevelFileName):
    with open(highLevelFileName, newline='') as highLevelFile:
        subfiles = highLevelFile.readlines()
        return [subfile.strip() for subfile in subfiles]

def getFilesWithTargetClasses(filesListFileName, targetClass1, targetClass2, minPoints):
    allFiles = loadEvalFilesFromFile(filesListFileName)
    # semanticClassGroupList, semanticClassGroupDict = getSemanticClassGroups()
    semanticClassGroupList, semanticClassGroupDict = getCondensedSemanticClassGroups()

    filesByClassGroup = [[] for i in range(len(semanticClassGroupList))]

    for fileName in allFiles:
        semanticClassId = extractSemanticClassFromFileName(fileName)
        classGroupIndex = semanticClassGroupDict[semanticClassId]
        filesByClassGroup[classGroupIndex].append(fileName)

    secondFilesSet = None
    if (targetClass1):
        targetClass1GroupIndex = semanticClassGroupDict[targetClass1]
        firstFilesSet = filesByClassGroup[targetClass1GroupIndex]
        if (targetClass2):
            targetClass2GroupIndex = semanticClassGroupDict[targetClass2]
            if (targetClass1GroupIndex != targetClass2GroupIndex):
                secondFilesSet = filesByClassGroup[targetClass2GroupIndex]
    else:
        firstFilesSet = allFiles

    filteredFirstFiles, _ = getFilesOverMinPoints(firstFilesSet, minPoints)
    filteredSecondFiles = None
    if (secondFilesSet):
        filteredSecondFiles, _ = getFilesOverMinPoints(secondFilesSet, minPoints)

    return filteredFirstFiles, filteredSecondFiles

def argParser():
    parser = argparse.ArgumentParser("./visualize_semantic_kitti_clusters.py")
    parser.add_argument(
        '--files_list_file', '-f',
        type=str,
        required=True,
        help='File name containing point cloud files and the representation for the point cloud',
    )

    parser.add_argument(
        '--target_class_1', '-t',
        type=int,
        required=False,
        help='Numeric id from semantic kitti of the class that we should be computing results for',
    )

    parser.add_argument(
        '--target_class_2', '-s',
        type=int,
        required=False,
        help='Numeric id from semantic kitti of the class that we should be computing results for',
    )

    parser.add_argument(
        '--min_points', '-m',
        type=int,
        required=False,
        default=0,
        help='Minimum number of points for files that should be displayed'
    )


    return parser.parse_known_args()


def displayPointCloud(pointCloudFile):
    semanticClassId = extractSemanticClassFromFileName(pointCloudFile)
    # classLabel = getLabelForClassNum(semanticClassId)
    classLabel = getLabelForCondensedClassNum(semanticClassId)
    pointCloudPointsWithFeats = np.load(pointCloudFile)
    pointCloudPoints = pointCloudPointsWithFeats[:, 0:3]
    pointCloudIntensityGray = pointCloudPointsWithFeats[:, 3]
    pointCloudIntensityRGB = np.column_stack((pointCloudIntensityGray, pointCloudIntensityGray, pointCloudIntensityGray))
    print(pointCloudIntensityRGB)
    print(pointCloudIntensityRGB.shape)
    open3dPoints = open3d.utility.Vector3dVector(pointCloudPoints)
    pointCloud = open3d.geometry.PointCloud(open3dPoints)
    pointCloud.colors = open3d.utility.Vector3dVector(pointCloudIntensityRGB)

    geometries = [pointCloud]
    open3d.visualization.draw_geometries(geometries, window_name=(classLabel + '_' + str(pointCloudPoints.shape[0])), width=970)

def displayPointCloudsList(pointCloudFilesList):
    for pointCloudFile in pointCloudFilesList:
        displayPointCloud(pointCloudFile)

if __name__ == "__main__":
    arguments, _ = argParser()

    firstClassFiles, secondClassFiles = getFilesWithTargetClasses(arguments.files_list_file, arguments.target_class_1,
                                                                  arguments.target_class_2, arguments.min_points)

    print("Displaying files")
    if (secondClassFiles):
        firstClassProcess = Process(target=displayPointCloudsList, args=(firstClassFiles,))
        secondClassProcess = Process(target=displayPointCloudsList, args=(secondClassFiles,))
        firstClassProcess.start()
        secondClassProcess.start()
        firstClassProcess.join()
        secondClassProcess.join()
    else:
        displayPointCloudsList(firstClassFiles)

    # Get up to 2 semantic class groups
    # min points
    # Get list of files OR read from directory (for now just doing list of files)


