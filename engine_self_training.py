import math
import sys
from typing import Iterable
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from timm.utils import accuracy

def train_one_epoch(model: torch.nn.Module, args, train_config,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, amp_autocast,
                    device: torch.device, epoch: int, loss_scaler, 
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, model_ema=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, ((images_weak, _, _), targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + step
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None: 
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]

        # ramp-up ema decay 
        model_ema.decay = train_config['model_ema_decay_init'] + (args.model_ema_decay - train_config['model_ema_decay_init']) * min(1, it/train_config['warm_it'])
        metric_logger.update(ema_decay=model_ema.decay)
        
        images_weak = images_weak.to(device, non_blocking=True)
        # mask = mask.to(device, non_blocking=True) 
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            conf_ratio=1
            pseudo_label_acc=1
            metric_logger.update(conf_ratio=conf_ratio)
            metric_logger.update(pseudo_label_acc=pseudo_label_acc)

        with amp_autocast():    

            logits = model(images_weak)

            if args.shots > 0 :
                loss_st = F.cross_entropy(logits, targets)
            # probs = F.softmax(logits,dim=-1)
            # probs_all = probs
            # # probs_all = utils.all_gather_with_grad(probs)
            # probs_batch_avg = probs_all.mean(0) # average prediction probability across all gpus
            loss = loss_st
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        
        if loss_scaler is not None:
            grad_norm = loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(grad_norm=grad_norm)
        else:                   
            loss.backward(create_graph=False)       
            optimizer.step()

        model_ema.update(model)
        torch.cuda.synchronize()  

        metric_logger.update(loss_st=loss_st.item())
        # metric_logger.update(loss_fair=loss_fair.item())
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        if log_writer is not None:            
            log_writer.update(loss_st=loss_st.item(), head="train")
            # log_writer.update(loss_fair=loss_fair.item(), head="train")
    
            log_writer.update(conf_ratio=conf_ratio, head="train")
            log_writer.update(pseudo_label_acc=pseudo_label_acc, head="train")          
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
def train_one_epoch_liftncd(model: torch.nn.Module, args, train_config,
                    data_loader: Iterable,data_loader_u: Iterable, optimizer: torch.optim.Optimizer, amp_autocast,
                    device: torch.device, epoch: int, loss_scaler, 
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, model_ema=None,thresholding_module=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    data_loader_iter = iter(data_loader)
    data_loader_iter_u = iter(data_loader_u)
    for step, (idx_u,((images_weak_u, images_strong_s, _), targets_u)) in enumerate(metric_logger.log_every(data_loader_u, print_freq, header)):
        try:
            (images_weak, _, _), targets = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(data_loader)
            (images_weak, _, _), targets = next(data_loader_iter)
        except TypeError:
            assert data_loader is None
            return None
        # assign learning rate for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None: 
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]

        # ramp-up ema decay 
        model_ema.decay = train_config['model_ema_decay_init'] + (args.model_ema_decay - train_config['model_ema_decay_init']) * min(1, it/train_config['warm_it'])
        metric_logger.update(ema_decay=model_ema.decay)
        idx_u = idx_u.to(device)
        images_weak, images_weak_u, images_strong_s = images_weak.to(device, non_blocking=True), images_weak_u.to(device, non_blocking=True), images_strong_s.to(device, non_blocking=True)
        # mask = mask.to(device, non_blocking=True) 
        targets = targets.to(device, non_blocking=True)
        targets_u = targets_u.to(device, non_blocking=True)
        

        num_lb = images_weak.shape[0]
        inputs = torch.cat((images_weak, images_strong_s))

        with torch.no_grad():             
            # pseudo-label with ema model
            _,probs_ema_new,_ = model_ema.ema(images_weak_u, ncd=True)
            # logits_base,logits_new,_ = model(images_weak_u, ncd=True)
            probs_ema = F.softmax(probs_ema_new,dim=-1)
            score, pseudo_targets = probs_ema.max(-1)


            dynamic_threshold = thresholding_module.get_threshold(pseudo_targets)
            # conf_mask = score>train_config['conf_threshold']
            conf_mask = score > dynamic_threshold
            # mask used for updating learning status
            selected_mask = (score > train_config['conf_threshold']).long()
            thresholding_module.update(idx_u, selected_mask, pseudo_targets)
            pseudo_label_acc = (pseudo_targets[conf_mask] == targets_u[conf_mask]).float().mean().item()           
            conf_ratio = conf_mask.float().sum()/conf_mask.size(0)
            metric_logger.update(conf_ratio=conf_ratio)
            metric_logger.update(pseudo_label_acc=pseudo_label_acc)
            
        with amp_autocast():    

            logits_base,logits_new,_ = model(inputs, ncd=True)
                
            # self-training loss
            loss_st = F.cross_entropy(logits_base[:num_lb], targets)
            loss_u = F.cross_entropy(logits_new[num_lb:][conf_mask], pseudo_targets[conf_mask])
            # loss_st = F.cross_entropy(logits[:num_lb][conf_mask], pseudo_targets[conf_mask])

            loss = loss_st + loss_u



        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        
        if loss_scaler is not None:
            grad_norm = loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(grad_norm=grad_norm)
        else:                   
            loss.backward(create_graph=False)       
            optimizer.step()

        model_ema.update(model)
        torch.cuda.synchronize()  

        metric_logger.update(loss_st=loss_st.item())
        # metric_logger.update(loss_fair=loss_fair.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        if log_writer is not None:            
            log_writer.update(loss_st=loss_st.item(), head="train")
            # log_writer.update(loss_fair=loss_fair.item(), head="train")

            log_writer.update(conf_ratio=conf_ratio, head="train")
            log_writer.update(pseudo_label_acc=pseudo_label_acc, head="train")          
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, model_ema=None, args=None):
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    if model_ema is not None:
        model_ema.ema.eval()   

    if args.dataset in ['other']:
        all_outputs = []
        all_ema_outputs = []
        all_targets = []
        
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)

        # compute output
        output = model(images)

        if args.dataset in ['other']:
            all_outputs.append(output.cpu())
            all_targets.append(target.cpu())   
        else:
            acc = accuracy(output, target)[0]
            metric_logger.meters['acc1'].update(acc.item(), n=images.shape[0])
        
        if model_ema is not None:
            ema_output = model_ema.ema(images)
            if args.dataset in ['other']:
                all_ema_outputs.append(ema_output.cpu())
            else:  
                ema_acc1 = accuracy(ema_output, target)[0]  
                metric_logger.meters['ema_acc1'].update(ema_acc1.item(), n=images.shape[0])

    # if args.dataset in ['imagenet', 'sun397']:
    if args.dataset in ['other']:
        mean_per_class,every_class = utils.mean_per_class(torch.cat(all_outputs), torch.cat(all_targets))
        metric_logger.meters['acc1'].update(mean_per_class) 
        # metric_logger.meters['acc_every'].update(every_class)
        if model_ema is not None:
            mean_per_class,every_class = utils.mean_per_class(torch.cat(all_ema_outputs), torch.cat(all_targets))
            metric_logger.meters['ema_acc1'].update(mean_per_class) 
            # metric_logger.meters['acc_every'].update(every_class)
            
    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1))    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_ncd(data_loader_base, data_loader_new, model, device, model_ema=None, args=None):
    
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if model_ema is not None:
        model_ema.ema.eval()
    if args.dataset in ['imagenet']:
        all_outputs = []
        all_ema_outputs = []
        all_targets = []
    for batch in metric_logger.log_every(data_loader_base, 10, header):
        images = batch[0].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)

        # compute output
        output_base,_,_ = model(images, ncd=True)

        if args.dataset in ['imagenet']:
            all_outputs.append(output_base.cpu())
            all_targets.append(target.cpu())   
        else:    
            acc = accuracy(output_base, target)[0]
            metric_logger.meters['acc1_base'].update(acc.item(), n=images.shape[0])
        
        if model_ema is not None:
            ema_output_base,_,_ = model_ema.ema(images, ncd=True) 
            
            # if args.dataset in ['pets', 'caltech101', 'ucf101']:
            if args.dataset in ['imagenet']:
                all_ema_outputs.append(ema_output_base.cpu())
            else:  
                ema_acc1 = accuracy(ema_output_base, target)[0]  
                metric_logger.meters['ema_acc1_base'].update(ema_acc1.item(), n=images.shape[0])

    if args.dataset in ['imagenet']:
        mean_per_class,every_class = utils.mean_per_class(torch.cat(all_outputs), torch.cat(all_targets))
        metric_logger.meters['acc1_base'].update(mean_per_class) 
        # metric_logger.meters['acc_every'].update(every_class)
        if model_ema is not None:
            mean_per_class,every_class = utils.mean_per_class(torch.cat(all_ema_outputs), torch.cat(all_targets))
            metric_logger.meters['ema_acc1_base'].update(mean_per_class) 
            # metric_logger.meters['acc_every'].update(every_class)

    if args.dataset in ['imagenet']:
        all_outputs = []
        all_ema_outputs = []
        all_targets = []
    all_ema_outputs = []
    all_targets = []
    for batch in metric_logger.log_every(data_loader_new, 10, header):
        images = batch[0].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)

        # compute output
        _,output_new,_ = model(images, ncd=True)
        
        # if args.dataset in ['pets', 'caltech101', 'ucf101']:
        if args.dataset in ['imagenet']:
            all_outputs.append(output_new.cpu())
            all_targets.append(target.cpu())   
        else:    
            acc = accuracy(output_new, target)[0]
            metric_logger.meters['acc1_new'].update(acc.item(), n=images.shape[0])
        
        if model_ema is not None:
            _,ema_output_new,_ = model_ema.ema(images, ncd=True) 
            
            # if args.dataset in ['pets', 'caltech101', 'ucf101']:
            if args.dataset in ['imagenet']:
                all_ema_outputs.append(ema_output_new.cpu())
            else:  
                all_ema_outputs.append(ema_output_new.cpu())
                all_targets.append(target.cpu())  
                ema_acc1 = accuracy(ema_output_new, target)[0]
                metric_logger.meters['ema_acc1_new'].update(ema_acc1.item(), n=images.shape[0])
    mean_per_class,every_class = utils.mean_per_class(torch.cat(all_ema_outputs), torch.cat(all_targets))

    if args.dataset in ['imagenet']:
        mean_per_class,every_class = utils.mean_per_class(torch.cat(all_outputs), torch.cat(all_targets))
        metric_logger.meters['acc1_new'].update(mean_per_class)
        if model_ema is not None:
            mean_per_class,every_class = utils.mean_per_class(torch.cat(all_ema_outputs), torch.cat(all_targets))
            metric_logger.meters['ema_acc1_new'].update(mean_per_class)
            # metric_logger.meters['acc_every'].update(every_class)

    print('* Acc_base@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1_base))    
    print('* Acc_new@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1_new))
    print(mean_per_class)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
