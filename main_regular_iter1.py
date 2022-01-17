# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       main
   Project Name:    detector synthesis
   Author :         Hengrong LAN
   Date:            2019/3/1
   Device:          GTX1080Ti

   Note:            remove attn
-------------------------------------------------
   Change Activity:
                   2020/6/29:
-------------------------------------------------
"""
import logging.config
from tqdm import tqdm

from networks.model_cnn import Reg_net

from utils.visualizer import Visualizer
from skimage.measure import compare_ssim,compare_psnr
from utils.dataset_regular import ReconDataset
from utils.trick import *
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import scipy.stats as st
import scipy.io as scio
import click
import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# network options
# parser.add_argument('--num_classes', type=int, default=1, help='the number of classes')
# parser.add_argument('--in_channels', type=int, default=1, help='channel of Network input')
# parser.add_argument('--num_filters', type=int, default=32, help='number of filters for initial_conv')
parser.add_argument('--dataset_pathr', type=str, default='./20201208/', help='path of dataset')
parser.add_argument('--vis_env', type=str, default='model_based_iter1_regularization', help='visualization environment')
parser.add_argument('--save_path', type=str, default='checkpoint/', help='path of saved model')
parser.add_argument('--file_name', type=str, default='Reg_net_iter1_regularization.ckpt', help='file name of saved model')
parser.add_argument('--learning_rate', type=int, default=0.004, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size of training')
parser.add_argument('--test_batch', type=int, default=64, help='batch_size of testing')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--loadcp', type=bool, default=False, help='if load model')
parser.add_argument('--num_epochs', type=int, default=300, help='the number of epoches')

args = parser.parse_args()


logging.config.fileConfig("./logging.conf")

# create logger
log = logging.getLogger()

def calc_confidence_interval(samples, confidence_value=0.95):
    # samples should be a numpy array
    if type(samples) is list:
        samples = np.asarray(samples)
    assert isinstance(samples, np.ndarray), 'args: samples {} should be np.array'.format(samples)
    # print('Results List:', samples)
    stat_accu = st.t.interval(confidence_value, len(samples) - 1, loc=np.mean(samples), scale=st.sem(samples))
    center = (stat_accu[0] + stat_accu[1]) / 2
    deviation = (stat_accu[1] - stat_accu[0]) / 2
    return center, deviation

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_top=10):
        self.reset()
        _array = np.zeros(shape=(num_top)) + 0.01
        self.top_list = _array.tolist()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def top_update_calc(self, val):
        # update the lowest or NOT
        if val > self.top_list[0]:
            self.top_list[0] = val
            # [lowest, ..., highest]
            self.top_list.sort()
        # update mean, deviation
        mean, deviation = calc_confidence_interval(self.top_list)
        best = self.top_list[-1]
        return mean, deviation, best

# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



@click.command()
# @click.option('--gpu', default='2')
# @click.option('--batch_size', default=2)
def main():
    with torch.cuda.device(None):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################放入你的模型###################
        # model = AttnBottom(args).to(device)
        model = Reg_net(1)
        model = nn.DataParallel(model)
        model = model.to(device)

        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_ssim_meter = AverageMeter()
        train_psnr_meter = AverageMeter()
        train_ssim_top20 = AverageMeter(num_top=20)
        train_psnr_top20 = AverageMeter(num_top=20)
        test_ssim_meter = AverageMeter()
        test_psnr_meter = AverageMeter()
        test_ssim_top10 = AverageMeter(num_top=10)
        test_psnr_top10 = AverageMeter(num_top=10)

        vis = Visualizer(env=args.vis_env,port=5167)

        # source activate pytorch

        train_dataset = ReconDataset(args.dataset_pathr, train=True)
        test_dataset = ReconDataset(args.dataset_pathr, train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= args.test_batch, shuffle=True)


        
        criterionMSE = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        if  args.loadcp:
            checkpoint = torch.load(args.save_path+'latest_'+args.file_name)
            # model.load_state_dict(checkpoint['state_dict'])
            # start_epoch = checkpoint['epoch'] - 1
            # curr_lr = checkpoint['curr_lr']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('%s%d' % ('training from epoch:' , start_epoch))
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            args.learning_rate = checkpoint['curr_lr']

        cudnn.benchmark = True

        total_step = len(train_loader)

        best_metric = {'test_ssim': 0, 'test_psnr': 0}
        log.info('train image num: {}'.format(train_dataset.__len__()))
        log.info('val image num: {}'.format(test_dataset.__len__()))

        end = time.time()
        # patch = (1, img_height // 2 ** 4, img_width // 2 ** 4)

        for epoch in range(args.start_epoch,  args.num_epochs):
            for batch_idx, (rawdata, out_data,gradup,gradtv) in enumerate((train_loader)):
                rawdata = rawdata.to(device)
                out_data = out_data.to(device)
                gradup = gradup.to(device)
                gradtv = gradtv.to(device)


                outputs = model(rawdata,gradup)
                
                # L_pixel
                data_mse = 20 *criterionMSE(outputs, out_data)
                loss = data_mse

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss.item(), rawdata.size(0))
                #losses_data_mes.update(data_mse.item(), rawdata.size(0))
                #losses_pe.update(loss.item(), rawdata.size(0))
                #losses_bf.update(loss.item(), rawdata.size(0))

                ssim = compare_ssim(np.array(outputs.cpu().detach().squeeze()), np.array(out_data.cpu().detach().squeeze()))
                train_ssim_meter.update(ssim)
                psnr = compare_psnr(np.array(outputs.cpu().detach().squeeze()), np.array(out_data.cpu().detach().squeeze()),
                                    data_range=255)
                train_psnr_meter.update(psnr)

                # visualization and evaluation
                if (batch_idx + 1) % 5 == 0:
                   output_img = outputs.detach()
                   gt_image = out_data.detach()
                   input_img = rawdata.detach()
                   grad_img = gradup.detach()
                   vis.img(name='inputs', img_=15*input_img[0])
                   vis.img(name='ground truth',img_=15*gt_image[0])
                   vis.img(name='outputs', img_=15*output_img[0])
                   vis.img(name='img_grad', img_=15*grad_img[0])
                   
                   

                batch_time.update(time.time() - end)
                end = time.time()

            log.info(
                'Epoch [{}], Start [{}], Step [{}/{}], Loss: {:.4f}, Time [{batch_time.val:.3f}({batch_time.avg:.3f})]'
                    .format(epoch + 1, args.start_epoch, batch_idx + 1, total_step, loss.item(),
                            batch_time=batch_time))

            vis.plot_multi_win(
                dict(
                    losses_total=losses.val,
                    losses = losses.avg
                ))
            #vis.plot_multi_win(
            #    dict(
            #        losses = losses.avg,
            #        losses_pe=losses_pe.avg,
            #        losses_bf=losses_bf.avg))

            vis.plot_multi_win(dict(train_ssim=train_ssim_meter.avg, train_psnr=train_psnr_meter.avg))
            log.info('tain_ssim: {}, train_psnr: {}'.format(train_ssim_meter.avg, train_psnr_meter.avg))

            # Validata
            if epoch % 5 == 0:
                    with torch.no_grad():
                        for batch_idx, (rawdata, out_data,gradup,gradtv) in enumerate((test_loader)):
                            rawdata = rawdata.to(device)
                            out_data = out_data.to(device)
                            gradup = gradup.to(device)
                            gradtv = gradtv.to(device)


                            outputs = model(rawdata,gradup)

                            ssim = compare_ssim(np.array(outputs.cpu().squeeze()), np.array(out_data.cpu().squeeze()))
                            test_ssim_meter.update(ssim)
                            psnr = compare_psnr(np.array(outputs.cpu().squeeze()), np.array(out_data.cpu().squeeze()),
                                                data_range=255)
                            test_psnr_meter.update(psnr)
                   
                            if (batch_idx +1) % 2 ==0:
                               output_img = outputs.detach()
                               gt_image = out_data.detach()
                               input_img = rawdata.detach()
                               grad_img = gradup.detach()
                               vis.img(name='test_input',img_=15*input_img[0])                               
                               vis.img(name='test_gt',img_=15*gt_image[0])
                               vis.img(name='test_output', img_=15*output_img[0])
                               vis.img(name='test_grad', img_=15*grad_img[0])


                        vis.plot_multi_win(dict(
                            test_ssim=test_ssim_meter.avg,
                            test_psnr=test_psnr_meter.avg))
                        log.info('test_ssim: {}, test_psnr: {}'.format(test_ssim_meter.avg, test_psnr_meter.avg))

            # Decay learning rate
            if (epoch + 1) % 50 == 0:
                args.learning_rate /= 2
                update_lr(optimizer, args.learning_rate)

            torch.save({'epoch': epoch,
                        'model': model,
                        'optimizer': optimizer,
                        'curr_lr': args.learning_rate,
                        },
                       args.save_path+'latest_'+args.file_name
                       )

            if best_metric['test_ssim'] < test_ssim_meter.avg:
                torch.save({'epoch': epoch,
                            'model': model,
                            'optimizer': optimizer,
                            'curr_lr': args.learning_rate,
                            },
                           args.save_path+'best_'+args.file_name
                           )
                best_metric['test_ssim'] = test_ssim_meter.avg
                best_metric['test_psnr'] = test_psnr_meter.avg
            log.info('best_ssim: {}, best_psnr: {}'.format(best_metric['test_ssim'], best_metric['test_psnr']))



if __name__ == '__main__':
    main()
