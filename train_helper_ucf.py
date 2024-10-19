import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
from datetime import datetime
from datasets.crowd import UCF_CC_50
from models import vgg19
from losses.ot_loss import OT_Loss
from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils
from train_helper import train_collate

class TrainerWithKFold:
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        sub_dir = 'input-{}_wot-{}_wtv-{}_reg-{}_nIter-{}_normCood-{}'.format(
            args.crop_size, args.wot, args.wtv, args.reg, args.num_of_iter_in_ot, args.norm_cood)

        self.save_dir = os.path.join('ckpts', sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        self.logger = log_utils.get_logger(os.path.join(self.save_dir, 'train-{:s}.log'.format(time_str)))
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            self.logger.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("GPU is not available")

        downsample_ratio = 8
        if args.dataset.lower() == 'cc':
            self.dataset = UCF_CC_50(args.data_dir, args.crop_size, downsample_ratio)
        else:
            raise NotImplementedError

        self.model = vgg19()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            self.logger.info('loading pretrained model from ' + args.resume)
            checkpoint = torch.load(args.resume, self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
        else:
            self.logger.info('random initialization')

        self.ot_loss = OT_Loss(args.crop_size, downsample_ratio, args.norm_cood, self.device, args.num_of_iter_in_ot, args.reg)
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0

    def train(self):
        """Training with 5-fold cross-validation"""
        args = self.args
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            self.logger.info(f'Fold {fold+1}/5')
            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_collate)
            val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=args.num_workers)

            for epoch in range(self.start_epoch, args.max_epoch + 1):
                self.logger.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)
                self.epoch = epoch
                self.train_epoch(train_loader)
                if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                    val_mae, val_mse = self.val_epoch(val_loader)
                    self.logger.info(f"Fold {fold+1}, Epoch {epoch+1}, Val MAE: {val_mae}, Val MSE: {val_mse}")
                    fold_results.append((val_mae, val_mse))

        avg_mae = np.mean([result[0] for result in fold_results])
        avg_mse = np.mean([result[1] for result in fold_results])
        self.logger.info(f'Average MAE: {avg_mae}, Average MSE: {avg_mse}')
        

    def train_epoch(self, train_loader):
        epoch_ot_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()

        self.model.train()
        for step, (inputs, points, gt_discrete) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            gt_discrete = gt_discrete.to(self.device)

            outputs, outputs_normed = self.model(inputs)
            ot_loss, _, _ = self.ot_loss(outputs_normed, outputs, points)
            count_loss = self.mae(outputs.sum(1).sum(1).sum(1), torch.from_numpy(gd_count).float().to(self.device))
            loss = ot_loss + count_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred_count = torch.sum(outputs.view(inputs.size(0), -1), dim=1).detach().cpu().numpy()
            pred_err = pred_count - gd_count
            epoch_loss.update(loss.item(), inputs.size(0))
            epoch_mse.update(np.mean(pred_err * pred_err), inputs.size(0))
            epoch_mae.update(np.mean(abs(pred_err)), inputs.size(0))

        self.logger.info(f'Epoch {self.epoch} Train, Loss: {epoch_loss.get_avg()}, MSE: {epoch_mse.get_avg()}, MAE: {epoch_mae.get_avg()}')

    def val_epoch(self, val_loader):
        self.model.eval()
        epoch_res = []
        with torch.no_grad():
            for inputs, count, _ in val_loader:
                inputs = inputs.to(self.device)
                outputs, _ = self.model(inputs)
                
                # Sum the outputs to get the predicted count
                pred_count = torch.sum(outputs).item()

                # Ensure `count` represents the actual number of people, derived from the length of keypoints
                if isinstance(count, torch.Tensor):
                    gt_count = count.sum().item()  # The sum of all elements in `count` gives the number of people
                else:
                    gt_count = len(count)  # In case `count` is a list of keypoints

                # Calculate the error
                res = gt_count - pred_count
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        
        return mae, mse



    def save_model(self, epoch, is_best=False):
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, f'{epoch}_ckpt.tar')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dic,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        if is_best:
            self.logger.info(f"Saving best model at epoch {epoch}")
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))

