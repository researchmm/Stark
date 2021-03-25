import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *

"""Below packages should be different among different models"""
# config related
from lib.config.stark_st1.config import cfg, update_config_from_file
# network related
from lib.models.stark import build_starkst
# loss computation related
from lib.train.actors import STARKSActor


def run(settings):
    settings.description = 'Training STARK-ST (stage1)'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # Create network and actor
    net = build_starkst(cfg)

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")

    # Loss functions
    objective = {'giou': giou_loss, 'l1': l1_loss}

    loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}

    # Define the forward propagation (including loss computation)
    if cfg.TRAIN.DEEP_SUPERVISION:
        raise ValueError("Deep supervision is not supported now.")
    else:
        actor = STARKSActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
