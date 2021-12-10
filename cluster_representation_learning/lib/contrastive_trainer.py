
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.serialization import default_restore_location

import MinkowskiEngine as ME

from tensorboardX import SummaryWriter
from omegaconf import OmegaConf

from model.cluster_label_model import ClusterLabelModel
import lib.unoriginal.distributed as du
from lib.trainer_base import ClusterTrainer
from lib.unoriginal.timer import Timer, AverageMeter

class NPairLossClusterTrainer(ClusterTrainer):
    # See https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html for loss definition

    def computeLoss(self, posFeatA, posFeatB, negativeFeats):
        # Assume negative feats already stacked into a matrix with each column having 1 feature
        # Assume posFeatA are row vectors
        # posSampleMat = torch.cat((posFeatA, posFeatB), 0)
        # negPosDots = torch.mm(posSampleMat, negativeFeats)
        # negPosDotsExp = torch.exp(negPosDots)
        # print("Computing feat A dot with negs")
        # print(torch.unsqueeze(posFeatA, 0))
        # print(negativeFeats)
        negPosDotsA = torch.mm(torch.unsqueeze(posFeatA, 0), negativeFeats)
        # print("Computing feat B dot with negs")
        # print(negPosDotsA)
        negPosDotsB = torch.mm(torch.unsqueeze(posFeatB, 0), negativeFeats)
        # print(negPosDotsB)
        negPosDotsExpA = torch.exp(negPosDotsA)
        negPosDotsExpB = torch.exp(negPosDotsB)
        # print("Neg exp")
        # print(negPosDotsExpA)
        # print(negPosDotsExpB)

        posSampleDotExp = torch.exp(torch.dot(posFeatA, posFeatB))
        # print("Pos feats")
        # print(torch.dot(posFeatA, posFeatB))
        # print(posSampleDotExp)

        lossANegExpSum = torch.sum(negPosDotsExpA)
        lossBNegExpSum = torch.sum(negPosDotsExpB)
        # print("Neg sums")
        # print(lossANegExpSum)
        # print(lossBNegExpSum)
        # print("Loss quantities")
        lossAInside = torch.div(posSampleDotExp, torch.add(posSampleDotExp, lossANegExpSum))
        # print(lossAInside)
        lossBInside = torch.div(posSampleDotExp, torch.add(posSampleDotExp, lossBNegExpSum))
        # print(lossBInside)

        # May need to squeeze some of these... want result to be scalar
        loss = -torch.log(lossAInside) - torch.log(lossBInside)
        # print(loss)
        return loss


    def trainIter(self, data_loader_iter, timers):
        # print("HERE!")
        data_meter, data_timer, total_timer = timers
        self.optimizer.zero_grad()
        batch_loss = {
            'loss': 0.0,
        }
        data_time = 0
        total_timer.tic()

        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)
        modelInput = ME.SparseTensor(feats=input_dict['feats'].to(self.cur_device), coords=input_dict['coords'].to(self.cur_device))
        self.model = self.model.double()
        modelOut = self.model(modelInput)
        modelOut = modelOut.F
        # print(modelOut)
        # print(modelOut.shape)

        # Need to extract positive and negative sample pairs
        posCorrespondences = input_dict['posMatches'].to(self.cur_device)
        # Find all indices where value isn't -1
        # posCorrespondences = torch.Tensor([2, -1, 0]).long()
        # posCorrespondences = torch.Tensor([2, -1, -1]).long()
        evalIndices = torch.nonzero(posCorrespondences != -1, as_tuple=False).squeeze(dim=1)
        # print("Eval indices")
        # print(evalIndices)
        # Get entries corresponding to eval indices

        numPairs = evalIndices.size(dim=0)
        # print("Num pairs")
        # print(numPairs)

        negCorrespondences = input_dict['negMatches']
        # negCorrespondences = [[1], [0, 1]]

        loss = 0 # TODO Is this a fair way to initialize this?
        # TODO is there a way to do this without a for loop?
        for i in range(numPairs):
            # print(modelOut)
            # print("I: " + str(i))
            featAIndex = evalIndices[i].squeeze()
            # print("Feat A index " + str(featAIndex))
            featA = modelOut[featAIndex]
            # print(featA)
            featBIndex = posCorrespondences[featAIndex]
            # print("Feat B index " + str(featBIndex))
            featB = modelOut[featBIndex]
            # print(featB)

            # negFeatIndices = negCorrespondences[i]
            negFeatIndices = torch.tensor(negCorrespondences[featAIndex]).long().to(self.cur_device)
            # print("Neg indices")
            # print(negFeatIndices)
            negFeats = torch.index_select(modelOut, 0, negFeatIndices)
            # print("Neg feats")
            # print(negFeats)
            negFeats = negFeats.transpose(0, 1)
            # print(negFeats)

            # TODO Can we do this?
            loss += self.computeLoss(featA, featB, negFeats)

        # TODO Compute loss

        #print("Final loss")
        #print(loss)
        loss.backward()

        result = {"loss": loss}
        if self.config.misc.num_gpus > 1:
            result = du.scaled_all_reduce_dict(result, self.config.misc.num_gpus)
        batch_loss['loss'] += result["loss"].item()

        self.optimizer.step()


        torch.cuda.empty_cache()
        total_timer.toc()
        data_meter.update(data_time)
        return batch_loss, False

# TODO consider implementing InfoNCELoss


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = F.pairwise_distance(anchor, positive)
        #print("Distance positive " + str(distance_positive))
        distance_negative = F.pairwise_distance(anchor, negative)
        #print("Distance negative: " + str(distance_negative))
        losses = F.relu(distance_positive - distance_negative + self.margin)
        #print(losses)
        return losses.mean() if size_average else losses.sum()

class TripletLossTrainer(ClusterTrainer):

    def __init__(self, initial_model, config, data_loader):

        ClusterTrainer.__init__(self, initial_model, config, data_loader)
        self.loss_func = TripletLoss(config.trainer.margin)

    def trainIter(self, data_loader_iter, timers):
        # print("HERE!")
        data_meter, data_timer, total_timer = timers
        self.optimizer.zero_grad()
        batch_loss = {
            'loss': 0.0,
        }
        data_time = 0
        total_timer.tic()

        data_timer.tic()
        input_dict = data_loader_iter.next()
        data_time += data_timer.toc(average=False)
        modelInput = ME.SparseTensor(feats=input_dict['feats'].to(self.cur_device), coords=input_dict['coords'].to(self.cur_device))
        self.model = self.model.double()
        modelOut = self.model(modelInput)
        modelOut = modelOut.F
        # print(modelOut)
        # print(modelOut.shape)

        # Need to extract positive and negative sample pairs
        posCorrespondences = input_dict['posMatches'].to(self.cur_device)
        # Find all indices where value isn't -1
        # posCorrespondences = torch.Tensor([2, -1, 0]).long()
        # posCorrespondences = torch.Tensor([2, -1, -1]).long()
        evalIndices = torch.nonzero(posCorrespondences != -1, as_tuple=False).squeeze(dim=1)
        posSamplesA = torch.index_select(modelOut, 0, evalIndices)
        posSamplesBIndices = torch.index_select(posCorrespondences, 0, evalIndices)
        posSamplesB = torch.index_select(modelOut, 0, posSamplesBIndices)

        fileOfInterestPosA = "/robodata/aaadkins/derived_data/cluster_rep_data/semantic_kitti/05/sem_kitti_cluster_05_scan1744_inst9043978_semClass10_points.npy"
        fileOfInterestPosB = "/robodata/aaadkins/derived_data/cluster_rep_data/semantic_kitti/00/sem_kitti_cluster_00_scan3067_inst4718602_semClass10_points.npy"
        fileOfInterestNeg = "/robodata/aaadkins/derived_data/cluster_rep_data/semantic_kitti/00/sem_kitti_cluster_00_scan144_inst393246_semClass30_points.npy"

        fileNames = input_dict['fileNames']

        printFeats = False
        if ((fileOfInterestPosA in fileNames) and (fileOfInterestPosB in fileNames) and (fileOfInterestNeg in fileNames)):
            printFeats = True

        if (printFeats):
            indexPosA = fileNames.index(fileOfInterestPosA)
            indexPosB = fileNames.index(fileOfInterestPosB)
            indexNeg = fileNames.index(fileOfInterestNeg)
            featPosA = modelOut[indexPosA, :]
            featPosB = modelOut[indexPosB, :]
            featNeg = modelOut[indexNeg, :]
            logging.info("Feat Pos A for file " + fileOfInterestPosA)
            logging.info(featPosA)
            logging.info("Feat Pos B for file " + fileOfInterestPosB)
            logging.info(featPosB)
            logging.info("Feat Neg for file " + fileOfInterestNeg)
            logging.info(featNeg)

        # print("Eval indices")
        # print(evalIndices)
        # Get entries corresponding to eval indices

        numPairs = evalIndices.size(dim=0)
        # print("Num pairs")
        # print(numPairs)

        negCorrespondences = input_dict['negMatches']
        negCorrespondenceIndices = []
        for i in range(numPairs):
            featAIndex = evalIndices[i].squeeze()
            negCorrespondenceIndices.append(negCorrespondences[featAIndex][0])
        negCorrespondenceIndicesTensor = torch.tensor(negCorrespondenceIndices).long().to(self.cur_device)
        negFeats = torch.index_select(modelOut, 0, negCorrespondenceIndicesTensor)

        loss = self.loss_func(posSamplesA, posSamplesB, negFeats)
        #print("Loss ")
        #print(loss)
        # negCorrespondences = [[1], [0, 1]]

        #loss = 0 # TODO Is this a fair way to initialize this?
        # TODO is there a way to do this without a for loop?
        #for i in range(numPairs):
            # print(modelOut)
            # print("I: " + str(i))
        #    featAIndex = evalIndices[i].squeeze()
            # print("Feat A index " + str(featAIndex))
        #    featA = modelOut[featAIndex]
            # print(featA)
        #    featBIndex = posCorrespondences[featAIndex]
            # print("Feat B index " + str(featBIndex))
        #    featB = modelOut[featBIndex]
            # print(featB)

            # negFeatIndices = negCorrespondences[i]
        #    negFeatIndex = torch.tensor(negCorrespondences[featAIndex][0]).long().to(self.cur_device)
        #    print("Neg index")
        #    print(negFeatIndex)
        #    negFeat = modelOut[negFeatIndex]
            # TODO can we do this in batch?
        #    loss += self.loss_func(featA, featB, negFeat)

        # TODO Compute loss

        #print("Final loss")
        #print(loss)
        loss.backward()

        result = {"loss": loss}
        if self.config.misc.num_gpus > 1:
            result = du.scaled_all_reduce_dict(result, self.config.misc.num_gpus)
        batch_loss['loss'] += result["loss"].item()

        self.optimizer.step()


        torch.cuda.empty_cache()
        total_timer.toc()
        data_meter.update(data_time)
        return batch_loss, printFeats

# TODO consider implementing InfoNCELoss


