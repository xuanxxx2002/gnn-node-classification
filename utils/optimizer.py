import torch.optim as optim


def build_optimizer(args, params):
    trainable = filter(lambda p: p.requires_grad, params)
    opt_map = {
        'adam':    lambda: optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay),
        'sgd':     lambda: optim.SGD(trainable,  lr=args.lr, momentum=0.95, weight_decay=args.weight_decay),
        'rmsprop': lambda: optim.RMSprop(trainable, lr=args.lr, weight_decay=args.weight_decay),
        'adagrad': lambda: optim.Adagrad(trainable, lr=args.lr, weight_decay=args.weight_decay),
    }
    optimizer = opt_map[args.opt]()

    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer
