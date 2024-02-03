import torch
from torch.autograd import Function
from torch import nn
import math
import numpy as np

class BatchCriterion(nn.Module):
    ''' Compute the loss within each batch  
    '''
    def __init__(self, T, M,batchSize):
        super(BatchCriterion, self).__init__()
        self.T = T
        self.M = M
        self.diag_mat = 1 - torch.eye(batchSize*2).cuda()
        
    def forward(self, x, targets):
        batchSize = x.size(0)
        
        #get positive innerproduct
        reordered_x = torch.cat((x.narrow(0,batchSize//2,batchSize//2),\
                x.narrow(0,0,batchSize//2)), 0)
        #reordered_x = reordered_x.data
        pos = ((x*reordered_x.data).sum(1) - self.M).div_(self.T).exp_()

        #get all innerproduct, remove diag
        all_prob = torch.mm(x,x.t().data).div_(self.T).exp_()*self.diag_mat
        all_div = all_prob.sum(1)

        lnPmt = torch.div(pos, all_div)

        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        loss = - (lnPmtsum)/batchSize
        return loss
