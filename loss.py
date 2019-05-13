import pdb

import torch
import torch.nn.functional as F

import numpy as np
def mixup_cross_entropy_loss(input_, target, size_average=True):

    #assert input.size() == target.size()
    #assert isinstance(input, Variable) and isinstance(target, Variable)
    N = min(input_.shape[1], target.shape[1])
    input_ = input_[:,0:N,:]
    target = target[:,0:N,:]
    return - (input_.log_softmax(dim=-1) * target).sum(dim=-1).mean()


def onehot(targets, num_classes):

    #assert isinstance(targets, torch.LongTensor)
    return torch.zeros(targets.shape[0], num_classes).scatter_(1, targets.view(-1, 1), 1)


def mixup(inputs, targets, num_classes, alpha=2):

    s = inputs.shape[0]
    gamma = np.random.beta(alpha, alpha)
    perm = torch.randperm(s)
    perm_input = inputs[perm]
    perm_target = targets[perm]

    return  inputs.mul_(gamma).add_(1 - gamma, perm_input), targets.mul_(gamma).add_(1 - gamma, perm_target)


def binary_cross_entropy(output, target):
    
    # Align the time_steps of output and target
    N = min(output.shape[1], target.shape[1])

    out = F.binary_cross_entropy(
        output[:, 0: N, :],
        target[:, 0: N, :]
    )

    return out


def mean_error(output, target, mask, loss_type='MSE'):

    # Align the time_steps of output and target
    N = min(output.shape[1], target.shape[1])

    output = output[:, 0: N, :]
    target = target[:, 0: N, :]
    mask = mask[:, 0: N ,:]

    normalize_value = torch.sum(mask)

    if loss_type == 'MAE':
        out = torch.sum(torch.abs(output - target) * mask) / normalize_value
    elif loss_type == 'MSE':
        out = torch.sqrt(torch.sum((output - target)**2 * mask) / normalize_value)

    return out


def hybrid_regr_loss(output_dict, target_dict, task_type, loss_type='MSE'):
    '''
    Hybrid loss for regression doa:
    Input:
        output_dict: predict dictionary
        target_dict: target dictionary
        task_type: 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
        loss_type: 'MSE' | 'MAE'
    Output:
        seld_loss: sed loss plus doa loss
        sed_loss: sed loss
        doa_loss: doa loss
    '''

    class_num = target_dict['events'].shape[-1]

    sed_loss = mixup_cross_entropy_loss(
        input_=output_dict['events'],
        target=target_dict['events']
    )

    azimuth_loss = mean_error(
        output=output_dict['doas'][:, :, :class_num],
        target=target_dict['doas'][:, :, :class_num],
        mask=target_dict['events'],
        loss_type=loss_type
    )

    elevation_loss = mean_error(
        output=output_dict['doas'][:, :, class_num:],
        target=target_dict['doas'][:, :, class_num:],
        mask=target_dict['events'],
        loss_type=loss_type
    )

    if task_type == 'sed_only':
        beta = 0
    elif task_type == 'two_staged_eval' or task_type == 'doa_only':
        beta = 1
    elif task_type == 'seld':
        beta = 0.2
    
    doa_loss = elevation_loss + azimuth_loss
    seld_loss = (1 - beta) * sed_loss + beta * doa_loss

    return seld_loss, sed_loss, doa_loss
