import math
import os.path as osp
import torch
import time
import pdb
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.losses import r1_penalty
from basicsr.metrics import calculate_metric
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from torchvision.ops import roi_align
from tqdm import tqdm
import sys
sys.path.append('.')
import numpy as np
import cv2
import torch.nn as nn
cv2.setNumThreads(1)
import torchvision
from torch.nn.parallel import DataParallel, DistributedDataParallel
from scripts.utils import pad_tensor_back
from thop import profile

@MODEL_REGISTRY.register()
class LighTDiff(BaseModel):

    def __init__(self, opt):
        super(LighTDiff, self).__init__(opt)

        # define u-net network
        self.unet = build_network(opt['network_unet'])
        self.unet = self.model_to_device(self.unet)
        opt['network_ddpm']['denoise_fn'] = self.unet 

        self.global_corrector = build_network(opt['network_global_corrector'])
        self.global_corrector = self.model_to_device(self.global_corrector)
        opt['network_ddpm']['network_global_corrector'] = self.global_corrector

        self.ddpm = build_network(opt['network_ddpm'])
        self.ddpm = self.model_to_device(self.ddpm)
        if isinstance(self.ddpm, (DataParallel, DistributedDataParallel)):
            self.bare_model = self.ddpm.module
        else:
            self.bare_model = self.ddpm

        self.bare_model.set_new_noise_schedule(schedule_opt=opt['ddpm_schedule'],
                                            device=self.device)
        self.bare_model.set_loss(device=self.device)
        self.print_network(self.ddpm)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.ddpm, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        if 'metrics' in self.opt['val'] and 'lpips' in self.opt['val']['metrics']:
            import lpips
            self.lpips = lpips.LPIPS(net='alex')
            self.lpips = self.model_to_device(self.lpips)
            if isinstance(self.lpips, (DataParallel, DistributedDataParallel)):
                self.lpips_bare_model = self.lpips.module
            else:
                self.lpips_bare_model = self.lpips



        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.ddpm.train()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # ----------- optimizer g ----------- #
        net_g_reg_ratio = 1
        normal_params = []
        logger = get_root_logger()
        for _, param in self.ddpm.named_parameters():
            if self.opt['train'].get('frozen_denoise', False):
                if 'denoise' in _:
                    logger.info(f'frozen {_}')
                    continue
            normal_params.append(param)
        optim_params_g = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim_g']['lr']
        }]
        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_g_reg_ratio # 1e-4
        lr = float(lr)
        betas = (0**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas)
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.LR = data['LR'].to(self.device)
        self.HR = data['HR'].to(self.device)
        if 'pad_left' in data:
            self.pad_left = data['pad_left'].to(self.device)
            self.pad_right = data['pad_right'].to(self.device)
            self.pad_top = data['pad_top'].to(self.device)
            self.pad_bottom = data['pad_bottom'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        pred_noise, noise, x_recon_cs, x_start, t, color_scale = self.ddpm(self.HR, self.LR, 
                  train_type=self.opt['train'].get('train_type', None),
                  different_t_in_one_batch=self.opt['train'].get('different_t_in_one_batch', None),
                  t_sample_type=self.opt['train'].get('t_sample_type', None),
                  pred_type=self.opt['train'].get('pred_type', None),
                  clip_noise=self.opt['train'].get('clip_noise', None),
                  color_shift=self.opt['train'].get('color_shift', None),
                  color_shift_with_schedule= self.opt['train'].get('color_shift_with_schedule', None),
                  t_range=self.opt['train'].get('t_range', None),
                  cs_on_shift=self.opt['train'].get('cs_on_shift', None),
                  cs_shift_range=self.opt['train'].get('cs_shift_range', None),
                  t_border=self.opt['train'].get('t_border', None),
                  down_uniform=self.opt['train'].get('down_uniform', False),
                  down_hw_split=self.opt['train'].get('down_hw_split', False),
                  pad_after_crop=self.opt['train'].get('pad_after_crop', False),
                  input_mode=self.opt['train'].get('input_mode', None),
                  crop_size=self.opt['train'].get('crop_size', None),
                  divide=self.opt['train'].get('divide', None),
                  frozen_denoise=self.opt['train'].get('frozen_denoise', None),
                  cs_independent=self.opt['train'].get('cs_independent', None),
                  shift_x_recon_detach=self.opt['train'].get('shift_x_recon_detach', None))
        if self.opt['train'].get('vis_train', False) and current_iter <= self.opt['train'].get('vis_num', 100) and \
            self.opt['rank'] == 0:

            save_img_path = osp.join(self.opt['path']['visualization'], 'train',
                                            f'{current_iter}_noise_level_{self.bare_model.t}.png')
            x_recon_print = tensor2img(self.bare_model.x_recon, min_max=(-1, 1))
            noise_print = tensor2img(self.bare_model.noise, min_max=(-1, 1))
            pred_noise_print = tensor2img(self.bare_model.pred_noise, min_max=(-1, 1))
            x_start_print = tensor2img(self.bare_model.x_start, min_max=(-1, 1))
            x_noisy_print = tensor2img(self.bare_model.x_noisy, min_max=(-1, 1))

            img_print  = np.concatenate([x_start_print, noise_print, x_noisy_print, pred_noise_print, x_recon_print], axis=1)
            imwrite(img_print, save_img_path)
        l_g_total = 0
        loss_dict = OrderedDict()

        l_g_x0 = F.smooth_l1_loss(x_recon_cs, x_start, beta = 0.5) * self.opt['train'].get('l_g_x0_w', 1.0)
        if self.opt['train'].get('gamma_limit_train', None) and color_scale <= self.opt['train'].get('gamma_limit_train', None):
            l_g_x0 = l_g_x0 * 0.5
        loss_dict['l_g_x0'] = l_g_x0
        l_g_total += l_g_x0

        if not self.opt['train'].get('frozen_denoise', False):
            l_g_noise = F.smooth_l1_loss(pred_noise, noise)
            loss_dict['l_g_noise'] = l_g_noise
            l_g_total += l_g_noise

        l_g_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        with torch.no_grad():
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            self.bare_model.eval()
            self.output = self.bare_model.ddim_LighT_sample(self.LR, 
                                                structure=self.opt['val'].get('structure'),
                                                continous=self.opt['val'].get('ret_process', False), 
                                                ddim_timesteps = self.opt['val'].get('ddim_timesteps', 50),
                                                return_pred_noise=self.opt['val'].get('return_pred_noise', False),
                                                return_x_recon=self.opt['val'].get('ret_x_recon', False),
                                                ddim_discr_method=self.opt['val'].get('ddim_discr_method', 'uniform'),
                                                ddim_eta=self.opt['val'].get('ddim_eta', 0.0),
                                                pred_type=self.opt['val'].get('pred_type', 'noise'),
                                                clip_noise=self.opt['val'].get('clip_noise', False),
                                                save_noise=self.opt['val'].get('save_noise', False),
                                                color_gamma=self.opt['val'].get('color_gamma', None),
                                                color_times=self.opt['val'].get('color_times', 1),
                                                return_all=self.opt['val'].get('ret_all', False),
                                                fine_diffV2=self.opt['val'].get('fine_diffV2', False),
                                                fine_diffV2_st=self.opt['val'].get('fine_diffV2_st', 200),
                                                fine_diffV2_num_timesteps=self.opt['val'].get('fine_diffV2_num_timesteps', 20),
                                                do_some_global_deg=self.opt['val'].get('do_some_global_deg', False),
                                                use_up_v2=self.opt['val'].get('use_up_v2', False))
            self.bare_model.train()
            
            if hasattr(self, 'pad_left') and not self.opt['val'].get('ret_process', False):
                self.output = pad_tensor_back(self.output, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.LR = pad_tensor_back(self.LR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
                self.HR = pad_tensor_back(self.HR, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom)
            return 

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def find_lol_dataset(self, name):
        if name[0] == 'r':
            return 'SYNC'
        elif name[0] == 'n' or name[0] == 'l':
            return 'REAL'
        else:
            return 'LOL'

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if self.opt['val'].get('fix_seed', False):
            next_seed = np.random.randint(10000000)
            logger = get_root_logger()
            logger.info(f'next_seed={next_seed}')
        if self.opt['val'].get('ret_process', False):
            with_metrics = False
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        metric_data = dict()
        metric_data_pytorch = dict()
        pbar = tqdm(total=len(dataloader), unit='image')
        if self.opt['val'].get('split_log', False):
            self.split_results = {}
            self.split_results['SYNC'] = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self.split_results['REAL'] = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self.split_results['LOL'] = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        total_time = 0
        for idx, val_data in enumerate(dataloader):
            if self.opt['val'].get('fix_seed', False):
                from basicsr.utils import set_random_seed
                set_random_seed(0)
            if not self.opt['val'].get('cal_all', False) and \
               not self.opt['val'].get('cal_score', False) and \
               int(self.opt['ddpm_schedule']['n_timestep']) >= 4 and idx >= 3:
                break
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['sr']], min_max=(-1, 1))
            gt_img = tensor2img([visuals['gt']], min_max=(-1, 1))
            lq_img = tensor2img([visuals['lq']], min_max=(-1, 1))
            if self.opt['val'].get('use_kind_align', False):
                gt_mean = np.mean(gt_img)
                sr_mean = np.mean(sr_img)
                sr_img = sr_img * gt_mean / sr_mean
                sr_img = np.clip(sr_img, 0, 255)
                sr_img = sr_img.astype('uint8')

            metric_data['img'] = sr_img
            metric_data['img2'] = gt_img
            metric_data_pytorch['img'] = self.output
            metric_data_pytorch['img2'] = self.HR
            path = val_data['lq_path'][0]
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                # print(save_img_path)
                if idx < self.opt['val'].get('show_num', 3) or self.opt['val'].get('show_all', False):
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                            f'{img_name}_{current_iter}.png')
                    if not self.opt['val'].get('ret_process', False):
                        if self.opt['val'].get('only_save_sr', False):
                            save_img_path = osp.join(self.opt['path']['visualization'],
                                            f'{img_name}.png')
                            imwrite(sr_img, save_img_path)
                        else:
                            imwrite(np.concatenate([lq_img, sr_img, gt_img], axis=1), save_img_path)
                    else:
                        imwrite(sr_img, save_img_path)
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if 'lpips' in opt_['type']:
                        opt_['device'] = self.device
                        opt_['model'] = self.lpips_bare_model
                    if 'pytorch' in opt_['type']:
                        if self.opt['val'].get('split_log', False):
                            self.split_results[self.find_lol_dataset(img_name)][name] += calculate_metric(metric_data_pytorch, opt_).item()
                        self.metric_results[name] += calculate_metric(metric_data_pytorch, opt_).item()
                    else:
                        if self.opt['val'].get('split_log', False):
                            self.split_results[self.find_lol_dataset(img_name)][name] += calculate_metric(metric_data, opt_)
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            
            torch.cuda.empty_cache()
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            if self.opt['val'].get('cal_score_num', None):
                if idx >= self.opt['val'].get('cal_score_num', None):
                    break


        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        if self.opt['val'].get('cal_score', False):
            import sys
            sys.exit()
        if self.opt['val'].get('fix_seed', False):
            from basicsr.utils import set_random_seed
            set_random_seed(next_seed)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        logger = get_root_logger()
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger.info(log_str)
        if self.opt['val'].get('split_log', False):
            for dataset_name, num in zip(['LOL', 'REAL', 'SYNC'], [15, 100, 100]):
                log_str = f'Validation {dataset_name}\n'
                for metric, value in self.split_results[dataset_name].items():
                    log_str += f'\t # {metric}: {value/num:.4f}\n'
                logger.info(log_str)
        
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if self.LR.shape != self.output.shape:
            self.LR = F.interpolate(self.LR, self.output.shape[2:])
            self.HR = F.interpolate(self.HR, self.output.shape[2:])
        out_dict['gt'] = self.HR.detach().cpu()
        out_dict['sr'] = self.output.detach().cpu()
        out_dict['lq'] = self.LR[:, :3, :, :].detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network([self.ddpm], 'net_g', current_iter, param_key=['params'])
