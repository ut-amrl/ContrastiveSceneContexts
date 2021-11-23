import os
import numpy as np
import random
import csv

def generateTestPointCloudFile(numPoints, fileName):
    pointCloud = np.random.rand(numPoints, 4)
    np.save(fileName, pointCloud)

def main():
    directory = "/home/amanda/Documents/scratch"
    pointCloudFileNames = []
    for i in range(10):
        pointCloudFileName = os.path.join(directory, "pointCloud_" + str(i) + ".npy")
        pointCloudFileNames.append(pointCloudFileName)
        generateTestPointCloudFile(random.randint(4, 8), pointCloudFileName)

    contrastivePairsBaseFileName = "contrastive_pairs.csv"
    with open(os.path.join(directory, contrastivePairsBaseFileName), 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        datawriter.writerow([pointCloudFileNames[0], pointCloudFileNames[2], pointCloudFileNames[4], pointCloudFileNames[5]])
        datawriter.writerow([pointCloudFileNames[6], pointCloudFileNames[1], pointCloudFileNames[2]])
        datawriter.writerow([pointCloudFileNames[3], pointCloudFileNames[7], pointCloudFileNames[6], pointCloudFileNames[8], pointCloudFileNames[9]])

if __name__ == "__main__":
    main()