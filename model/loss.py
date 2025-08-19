import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import namedtuple
#import torch.nn.functional as F

def BCELoss_ClassWeights(inputT, target, class_weights):
    """
    BCE loss with class weights
    :param inputT: (n, d)
    :param target: (n, d)
    :param class_weights: (d,)
    :return:
    """
    inputT = torch.clamp(inputT, min=1e-7, max=1 - 1e-7)
    w0, w1 = class_weights
    bce = - w0 * target * torch.log(inputT) - w1 * (1 - target) * torch.log(1 - inputT)
    # weighted_bce = (bce * class_weights).sum(axis=1) / class_weights.sum(axis=1)[0]
    # final_reduced_over_batch = weighted_bce.mean(axis=0)
    final_reduced_over_batch = bce.mean(axis=0)
    return final_reduced_over_batch


class BinaryCrossEntropy(torch.nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        if class_weights is None:
            self.bce = torch.nn.BCELoss()
        else:
            self.bce = partial(BCELoss_ClassWeights, class_weights=class_weights)

    def forward(self, y_pred, y_true):
        return self.bce(y_pred.flatten(), y_true.flatten())


def nll_loss(output, target):
    return F.nll_loss(output, target)

LossResult = namedtuple('LossResult', ['loss', 'bce_loss', 'l1_loss', 'balancing_loss'])

def BCELossLike_MoE(output, true, weights, class_weights):
    """
    BCE loss with class weights
    :param inputT: (n, d)
    :param target: (n, d)
    :param class_weights: (d,)
    :return:
    """
    batch_size = output.shape[0]
    class_weights_batch = torch.full((batch_size,1), class_weights[0])
    class_weights_batch[true == 1] = class_weights[1]
    #class_weights_batch = class_weights_batch.squeeze()
    true = 2 * true - 1  
    output_norm = torch.sigmoid(output * true.unsqueeze(1))
    loss = - torch.log( 1/output.shape[1] * torch.bmm( weights.unsqueeze(1), output_norm ) ) * class_weights_batch #/ (class_weights[0] + class_weights[1])
    loss = torch.mean(loss)
    return loss

def LoadBalancingLoss(gate_logits, lambda_balancing=0.0001):
    # Compute the average usage of each expert
    expert_usage = gate_logits.mean(dim=0)
    # Penalize deviation from uniform usage
    loss = torch.std(expert_usage)

    return lambda_balancing * loss

def L1Loss(model, lambda_l1_exp=0.005, lambda_l1_gate=0.1):
    l1_norm_exp = sum(p.abs().sum() for p in model.experts.parameters())
    l1_norm_gate = sum(p.abs().sum() for p in model.gate.parameters())

    return lambda_l1_exp * l1_norm_exp + lambda_l1_gate * l1_norm_gate

class MoELoss(torch.nn.Module):
    def __init__(self, lambda_l1_gate, lambda_l1_exp, lambda_balancing, class_weights=[1, 1]):
        super().__init__()
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.lambda_balancing = lambda_balancing
        self.lambda_l1_exp = lambda_l1_exp
        self.lambda_l1_gate = lambda_l1_gate
        self.bce = partial(BCELossLike_MoE, class_weights=class_weights)

    def forward(self, outputs, targets, weights, model, validation=False):
        #bce_loss = BCELossLike_MoE(outputs, targets, weights)
        bce_loss = self.bce(outputs, targets, weights)
        if self.lambda_balancing != 0:
            balancing_loss = LoadBalancingLoss(weights, self.lambda_balancing)
        else:
            balancing_loss = torch.tensor([0])
        if (self.lambda_l1_exp !=0 or self.lambda_l1_gate != 0) and (validation == False):
            l1_loss = L1Loss(model, self.lambda_l1_exp, self.lambda_l1_gate )
        else: 
            l1_loss = torch.tensor([0])

        loss = bce_loss + balancing_loss + l1_loss

        return LossResult(loss, bce_loss, l1_loss, balancing_loss)
