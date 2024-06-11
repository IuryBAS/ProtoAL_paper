'''
The code bellow are in larger part borrowed from the ProtoPNet code available in https://github.com/cfchen-duke/ProtoPNet

MIT License

Copyright (c) 2019 Chaofan Chen (cfchen-duke), Oscar Li (OscarcarLi),
Chaofan Tao, Alina Jade Barnett, Cynthia Rudin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.




Minor modifications that were made are pointed out in comments
'''

import torch


joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 10

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

def log_metric(func):
    def wrapper(*args, **kwargs):
        metric = func(*args, **kwargs)
        logger = 1
    return wrapper

def configure_optimizers(model, optim):
    if optim == 'warm':
        warm_optimizer_specs = \
        [{'params': model.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
         {'params': model.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
        ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
        return warm_optimizer
    elif optim == 'joint':
        joint_optimizer_specs = \
        [{'params': model.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
         {'params': model.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
         {'params': model.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
        ]
        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        # We replace the StepLR in the original ProtoPNet by an ExponentialLR as scheduler approach
        joint_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(joint_optimizer, gamma=0.99) 
        return joint_optimizer, joint_lr_scheduler
    elif optim == 'last':
        last_layer_optimizer_specs = [{'params': model.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
        return last_layer_optimizer

def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')





           

