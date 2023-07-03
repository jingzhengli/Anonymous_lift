import argparse
import datetime
import numpy as np
import time
import os
import os.path as osp
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
print(torch.cuda.current_device())
import torch.backends.cudnn as cudnn
import json
# import os
from contextlib import suppress
import random

from pathlib import Path
from collections import OrderedDict

from ema import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import ConfidenceBasedSelfTrainingLoss, DynamicThresholdingModule
# from build_dataset import build_dataset
from datasets import build_dataset,build_transform
from datasets.utils import build_data_loader

from engine_self_training import train_one_epoch_liftncd, evaluate_ncd
from dassl.utils import load_checkpoint
from model import clip_classifier

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def get_args():
    parser = argparse.ArgumentParser('MUST training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--eval_freq', default=1, type=int) 
    
    # CLIP parameters
    parser.add_argument("--template", default='templates.json', type=str)
    parser.add_argument("--classname", default='classes.json', type=str)
    parser.add_argument('--clip_model', default='ViT-B/16', help='pretrained clip model name') #ViT-B/16 RN50
    parser.add_argument('--image_mean', default=(0.48145466, 0.4578275, 0.40821073)) 
    parser.add_argument('--image_std', default=(0.26862954, 0.26130258, 0.27577711)) 
    parser.add_argument('--input_size', default=224, type=int, help='images input size') 
  
    # training parameters
    parser.add_argument("--train_config", default='train_configs.json', type=str, help='training configurations') 
    parser.add_argument('--mask', action='store_true')
    # parser.set_defaults(mask=True)
    parser.add_argument('--adapter', action='store_true')
    # parser.set_defaults(adapter=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9998, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.65)
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='epochs')
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--bce_type', type=str, default="cos")
    parser.add_argument('--momentum', default=0.999, type=float,
                        help='momentum coefficient for updating q_hat (default: 0.999)')

    # Augmentation parameters  
    parser.add_argument('--train_crop_min', default=0.3, type=float)
    parser.add_argument('--color_jitter', type=float, default=0, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int, help='number of the classification types')
    parser.add_argument('--dataset', default='ucf101', type=str, help='dataset name') #sun397 ucf101 pets dtd caltech101
    parser.add_argument('--root_path', default='./DATA/', type=str, help='dataset path')
    parser.add_argument('--shots', default='16', type=int, help='few shot')
    parser.add_argument('--subsample', default='base', type=str, help='base,new')
    
    parser.add_argument('--output_dir', default='', help='path to save checkpoint and log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    # parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--warmup', default=False, type=bool)
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.set_defaults(eval=True)
    parser.add_argument('--num_workers', default=10, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true')

    return parser.parse_args()


def main(args):
    # utils.init_distributed_mode(args)

    train_configs = json.load(open(args.train_config,'r'))
    train_config = train_configs[args.dataset+'_'+args.clip_model]
    
    if not args.output_dir:
        args.output_dir = os.path.join('output_liftncd',args.dataset)
        args.output_dir = os.path.join(args.output_dir, "adapter%s_shot%d_lr%s_tau%.1f_epoch%d_lr%.5f"%(args.adapter, args.shots,args.clip_model[:5],train_config['conf_threshold'],args.epochs, train_config['lr']))
        
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(args)

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    # create image classifier from pretrained clip model
    dataset_base = build_dataset(args.dataset,args.root_path,args.shots,"base") # cfg['dataset'], cfg['root_path'], cfg['shots'])
    dataset_new = build_dataset(args.dataset,args.root_path,-1,"new")
    args.classnames = dataset_base.classnames
    args.classnames_new = dataset_new.classnames
    model = clip_classifier(args)
    args.nb_classes = len(args.classnames)
    import torchvision.transforms as transforms
    test_tranform = transforms.Compose([
        transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(args.image_mean),
            std=torch.tensor(args.image_std))
    ])
    train_tranform = build_transform(is_train=True,args=args, train_config=train_config)

    data_loader_train_x = build_data_loader(data_source=dataset_base.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, drop_last=True)
    # data_loader_train_u = build_data_loader(data_source=dataset_new.train_u, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, drop_last=True,index=True)
    data_loader_train_u = build_data_loader(data_source=dataset_new.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, drop_last=True,index=True)
    data_loader_val_base = build_data_loader(data_source=dataset_base.test, batch_size=args.batch_size, is_train=False, tfm=test_tranform, shuffle=False, drop_last=False)
    data_loader_val_new = build_data_loader(data_source=dataset_new.test, batch_size=args.batch_size, is_train=False, tfm=test_tranform, shuffle=False, drop_last=False)

    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = utils.TensorboardLogger(log_dir=args.output_dir)
    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(dict(args._get_kwargs())) + "\n")

    model_ema = ModelEma(
        model,
        decay=args.model_ema_decay,
        resume='')
    print("Using EMA with decay = %.5f" % (args.model_ema_decay) )

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * utils.get_world_size()
    num_training_steps_per_epoch = len(data_loader_train_u)
    if args.adapter:
        args.lr = 0.001
        args.min_lr = 0.00001
    else:
        args.lr = train_config['lr'] * total_batch_size / 64
        args.min_lr = args.min_lr * total_batch_size / 64
    args.epochs = train_config['epochs']
    args.eval_freq = train_config['eval_freq']
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training examples = %d" % len(dataset_new.test))

    if args.clip_model == "RN50" or args.adapter:
        assigner = None
    else:
        num_layers = model_without_ddp.model.visual.transformer.layers
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    optimizer = create_optimizer(
        args, model_without_ddp,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    if args.amp:
        loss_scaler = NativeScaler()
        amp_autocast = torch.cuda.amp.autocast
    else:
        loss_scaler = None
        amp_autocast = suppress

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    thresholding_module = DynamicThresholdingModule(train_config['conf_threshold'], args.warmup, lambda x: x / (2 - x), len(dataset_new.classnames),
                                                    len(dataset_new.train_x), device=device)

    if args.eval:
        test_stats = evaluate_ncd(data_loader_val_base,data_loader_val_new, model, device, model_ema=model_ema, args=args)

        log_stats = {'zero-shot': test_stats['acc1_new']}
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
        print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch_liftncd(
            model, args, train_config,
            data_loader_train_x,data_loader_train_u, optimizer, amp_autocast, device, epoch, loss_scaler, 
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            model_ema=model_ema,
            thresholding_module=thresholding_module
        )
        if args.output_dir and utils.is_main_process() and (epoch + 1) % args.eval_freq == 0:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)

            test_stats = evaluate_ncd(data_loader_val_base,data_loader_val_new, model, device, model_ema=model_ema, args=args)
            print(f"Accuracy of the network on the {len(dataset_base.test)} test images: {test_stats['acc1_base']:.1f}%")
            print(f"Accuracy of the network on the {len(dataset_new.test)} test images: {test_stats['acc1_new']:.1f}%")
            if max_accuracy < test_stats["acc1_base"]:
                max_accuracy = test_stats["acc1_base"]
                if args.output_dir:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1_base'], head="test", step=epoch)
                log_writer.update(test_ema_acc1=test_stats['ema_acc1_base'], head="test", step=epoch)
                log_writer.flush()
                
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    opts = get_args()
    main(opts)
