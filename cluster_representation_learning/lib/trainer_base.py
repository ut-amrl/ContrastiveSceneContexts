import joblib
import logging
import os

import torch
import torch.optim as optim
from torch.serialization import default_restore_location

import copy

import MinkowskiEngine as ME

from tensorboardX import SummaryWriter
from omegaconf import OmegaConf

from model.cluster_label_model import ClusterLabelModel
import lib.unoriginal.distributed as du
from lib.unoriginal.timer import Timer, AverageMeter

from lib.test_data_loader import createDataLoader

from lib.data_loading_utils import loadPointCloudFromFile

def load_state(model, weights, lenient_weight_loading=False):
  if du.get_world_size() > 1:
      _model = model.module
  else:
      _model = model

  if lenient_weight_loading:
    model_state = _model.state_dict()
    filtered_weights = {
        k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
    }
    logging.info("Load weights:" + ', '.join(filtered_weights.keys()))
    weights = model_state
    weights.update(filtered_weights)

  _model.load_state_dict(weights, strict=True)



class ClusterTrainer:
    def __init__(self, initial_model, config, data_loader):
        assert config.misc.use_gpu and torch.cuda.is_available(), "DDP mode must support GPU"
        self.stat_freq = config.trainer.stat_freq
        self.lr_update_freq = config.trainer.lr_update_freq
        self.checkpoint_freq = config.trainer.checkpoint_freq
        self.config = config

        self.is_master = du.is_master_proc(config.misc.num_gpus) if config.misc.num_gpus > 1 else True

        # Model initialization
        self.cur_device = torch.cuda.current_device()
        model = initial_model

        model = model.cuda(device=self.cur_device)
        if config.misc.num_gpus > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[self.cur_device],
                output_device=self.cur_device,
                broadcast_buffers=False,
            )

        self.config = config
        self.model = model

        self.optimizer = getattr(optim, config.opt.optimizer)(
            model.parameters(),
            lr=config.opt.lr,
            momentum=config.opt.momentum,
            weight_decay=config.opt.weight_decay)


        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.opt.exp_gamma)
        self.curr_iter = 0
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader

        if (config.data.eval_dataset_file != ''):
            datafiles = self.loadEvalFilesFromFile(config.data.eval_dataset_file)
            self.eval_data_loader = createDataLoader(datafiles, config.data.voxel_size, config.data.batch_size)

        # ---------------- optional: resume checkpoint by given path ----------------------
        if config.net.finetuned_weights and os.path.isfile(config.net.finetuned_weights):
            if self.is_master:
                logging.info('===> Loading weights: ' + config.net.finetuned_weights)
            state = torch.load(config.net.finetuned_weights, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            load_state(model, state['state_dict'], config.misc.lenient_weight_loading)
            if self.is_master:
                logging.info('===> Loaded weights: ' + config.net.finetuned_weights)

        # ---------------- default: resume checkpoint in current folder ----------------------
        checkpoint_fn = 'weights/weights.pth'
        if os.path.isfile(checkpoint_fn):
            if self.is_master:
                logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            state = torch.load(checkpoint_fn, map_location=lambda s, l: default_restore_location(s, 'cpu'))
            self.curr_iter = state['curr_iter']
            load_state(model, state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            if self.is_master:
                logging.info("=> loaded checkpoint '{}' (curr_iter {})".format(checkpoint_fn, state['curr_iter']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(checkpoint_fn))

        if self.is_master:
            self.writer = SummaryWriter(logdir='logs')
            if not os.path.exists('weights'):
                os.makedirs('weights', mode=0o755)
            OmegaConf.save(config, 'config.yaml')

    def _save_checkpoint(self, curr_iter, filename='checkpoint'):
        if not self.is_master:
            return
        _model = self.model.module if du.get_world_size() > 1 else self.model
        state = {
            'curr_iter': curr_iter,
            'state_dict': _model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        filepath = os.path.join('weights', f'{filename}.pth')
        logging.info("Saving checkpoint: {} ...".format(filepath))
        torch.save(state, filepath)
        # Delete symlink if it exists
        if os.path.exists('weights/weights.pth'):
            os.remove('weights/weights.pth')
        # Create symlink
        os.system('ln -s {}.pth weights/weights.pth'.format(filename))

    def loadEvalFilesFromFile(self, highLevelFileName):
        with open(highLevelFileName, newline='') as highLevelFile:
            subfiles = highLevelFile.readlines()
            return [subfile.strip() for subfile in subfiles]

    def outputEvalResults(self, curr_iter):
        # if not self.is_master:
        #     return
        #
        # if self.config.data.eval_dataset_file == '':
        #     return

        fileOfInterestPosA = "/robodata/aaadkins/derived_data/cluster_rep_data/semantic_kitti/05/sem_kitti_cluster_05_scan1744_inst9043978_semClass10_points.npy"
        fileOfInterestPosB = "/robodata/aaadkins/derived_data/cluster_rep_data/semantic_kitti/00/sem_kitti_cluster_00_scan3067_inst4718602_semClass10_points.npy"
        fileOfInterestNeg = "/robodata/aaadkins/derived_data/cluster_rep_data/semantic_kitti/00/sem_kitti_cluster_00_scan144_inst393246_semClass30_points.npy"

        files = [fileOfInterestPosA, fileOfInterestPosB, fileOfInterestNeg]

        # outFileName = self.config.data.eval_dataset_file + "_result_" + str(curr_iter) + ".pkl"
        self.model.eval()

        # output = []

        # lastOutput = 0
        with torch.no_grad():
            for pointCloudFile in files:
            # for batch in self.eval_data_loader:
            #     coords = batch[0].double()
            #     feats = batch[1]
                coords, feats = loadPointCloudFromFile(pointCloudFile, self.voxelSize)
                coords = coords.double()
                tensor = ME.SparseTensor(coords=coords.to(self.cur_device), feats=feats.to(self.cur_device))


                outForBatch = self.model(tensor)

                outFeats = outForBatch.F

                logging.info("File " + pointCloudFile)
                logging.info(outFeats)
                # for i in range(outForBatch.shape[0]):
                #     output.append(outFeats[i, :].cpu().detach().numpy())
                # numProcessed = len(output)
                # if (numProcessed > lastOutput + 500):
                #     print("Processed " + str(numProcessed))
                #     lastOutput = numProcessed

        # datafiles = self.loadEvalFilesFromFile(self.config.data.eval_dataset_file)

        # if (len(output) != len(datafiles)):
        #     print("Not as many labels generated as test files. ")
        #     return
        # combinedResults = [(datafiles[i], output[i]) for i in range(len(datafiles))]
        # print("Results file: " + outFileName)
        # with open(outFileName, 'wb') as resultsFile:
        #     joblib.dump(combinedResults, resultsFile)

        # self.eval_data_loader = createDataLoader(datafiles, self.config.data.voxel_size, self.config.data.batch_size)
        self.model.train()


    def train(self):

        curr_iter = self.curr_iter
        data_loader_iter = self.data_loader.__iter__()
        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()

        while (curr_iter < self.config.opt.max_iter):

            curr_iter += 1
            epoch = curr_iter / len(self.data_loader)

            batch_loss, evalKeyFiles = self.trainIter(data_loader_iter, [data_meter, data_timer, total_timer])

            # update learning rate
            if curr_iter % self.lr_update_freq == 0 or curr_iter == 1:
                lr = self.scheduler.get_last_lr()
                self.scheduler.step()

            # Print logs
            if curr_iter % self.stat_freq == 0 and self.is_master:
                self.writer.add_scalar('train/loss', batch_loss['loss'], curr_iter)
                logging.info(
                    "Train Epoch: {:.3f} [{}/{}], Current Loss: {:.3e}"
                    .format(epoch, curr_iter,
                            len(self.data_loader), batch_loss['loss']) +
                    "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}, LR: {}".format(
                        data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg,
                        self.scheduler.get_last_lr()))
                data_meter.reset()
                total_timer.reset()

            # save checkpoint
            if self.is_master and curr_iter % self.checkpoint_freq == 0:
                lr = self.scheduler.get_last_lr()
                logging.info(f" Epoch: {epoch}, LR: {lr}")
                checkpoint_name = 'checkpoint'
                if not self.config.trainer.overwrite_checkpoint:
                    checkpoint_name += '_{}'.format(curr_iter)
                self._save_checkpoint(curr_iter, checkpoint_name)
            if (evalKeyFiles):
                self.outputEvalResults(curr_iter)

    def trainIter(self, data_loader_iter, timers):
        pass

