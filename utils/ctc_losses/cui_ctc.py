import copy
import numpy as np
import os
import pdb
import time
import torch as T
import torch.cuda
from torch.autograd import Variable

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

cuda = torch.cuda.is_available()

if cuda:
    floatX = T.cuda.FloatTensor
    intX = T.cuda.IntTensor
    byteX = T.cuda.ByteTensor
    longX = T.cuda.LongTensor
else:

    floatX = T.FloatTensor
    intX = T.IntTensor
    byteX = T.ByteTensor
    longX = T.LongTensor

'''

floatX = T.float32
intX = T.int
byteX = T.uint8
longX = T.long

'''

def m_eye(n, k=0):
    assert k < n and k >= 0
    if k == 0:
        '''## torch eye returns 2d tensor diagonal matrix of size n*n with ones in the diagonal'''
        return T.eye(n).type(floatX)
    else:
        ''' shift diagon of ones related to k'''
        return T.cat((T.cat((T.zeros(n-k, k), T.eye(n-k)), dim=1), T.zeros(k, n)), dim=0).type(floatX)



def log_sum_exp_axis(a, uniform_mask=None, dim=0):
    assert dim == 0
    eps_nan = -1e8
    eps = 1e-26
    _max = T.max(a, dim=dim)[0]

    if not uniform_mask is None:
        nz_mask2 = T.gt(a, eps_nan) * uniform_mask
        nz_mask1 = T.gt(_max, eps_nan) * T.ge(T.max(uniform_mask, dim=dim)[0], 1)
    else:
        nz_mask2 = T.gt(a, eps_nan)
        nz_mask1 = T.gt(_max, eps_nan)

    # a-max
    a = a - _max[None]

    # exp
    _exp_a = T.zeros_like(a).type(floatX)
    _exp_a[nz_mask2] = T.exp(a[nz_mask2])

    # sum exp
    _sum_exp_a = T.sum(_exp_a, dim=dim)

    out = T.ones_like(_max).type(floatX) * eps_nan
    out[nz_mask1] = T.log(_sum_exp_a[nz_mask1] + eps) + _max[nz_mask1]
    return out


















def log_batch_dot(alpha_t, rec):
    '''
    alpha_t: (batch, 2U+1)
    rec: (batch, 2U+1, 2U+1)
    '''
    eps_nan = -1e8
    # a+b
    #print("alpha {} red {}".format(alpha_t[:,:,None], rec.size()))
    _sum = alpha_t[:, :, None] + rec
    #print("sum {}".format(_sum))
    _max_sum = T.max(_sum, dim=1)[0]
    #print("max_sum {} \n  max_sum {} ".format(_max_sum,_max_sum.size()))
    nz_mask1 = T.gt(_max_sum, eps_nan) # max > eps_nan
    nz_mask2 = T.gt(_sum, eps_nan)     # item > eps_nan


    # a+b-max
    _sum = _sum- _max_sum[:, None]

    # exp
    _exp = T.zeros_like(_sum).type(floatX)
    _exp[nz_mask2] = T.exp(_sum[nz_mask2])

    # sum exp
    _sum_exp = T.sum(_exp, dim=1)

    out = T.ones_like(_max_sum).type(floatX) * eps_nan
    out[nz_mask1] = T.log(_sum_exp[nz_mask1]) + _max_sum[nz_mask1]
    #print(out)
    return out


def log_sum_exp(*arrs):
#    return T.max(a.clone(), b.clone()) + T.log1p(T.exp(-T.abs(a.clone()-b.clone())))
    c = T.cat(list(map(lambda x:x[None], arrs)), dim=0)
    return log_sum_exp_axis(c, dim=0)
























def ctc_loss_log(pred, pred_len, token, token_len, blank=0):
    '''
    :param pred: (Time, batch, voca_size+1)
    :param pred_len: (batch)
    :param token: (batch, U)
    :param token_len: (batch)
    '''
    Time, batch = pred.size(0), pred.size(1)
    U = token.size(1)
    eps_nan = -1e8

    # token_with_blank
    token_with_blank = T.cat((T.zeros(batch, U, 1).type(longX), token[:, :, None]), dim=2).view(batch, -1)    # (batch, 2U)
    token_with_blank = T.cat((token_with_blank, T.zeros(batch, 1).type(longX)), dim=1)  # (batch, 2U+1)
    length = token_with_blank.size(1)
    #print(length)
    #print("pred after arange size {}".format(pred.size()))
    pred = pred[T.arange(0, Time).type(longX)[:, None, None], T.arange(0, batch).type(longX)[None, :, None], token_with_blank[None, :]]  # (T, batch, 2U+1)
    #print("pred after arange size {}".format(pred.size()))
    # recurrence relation.foa
    #print(T.ne(token_with_blank[:, :-2], token_with_blank[:, 2:]))
    sec_diag = T.cat((T.zeros((batch, 2)).type(floatX), T.ne(token_with_blank[:, :-2], token_with_blank[:, 2:]).type(floatX)), dim=1) * T.ne(token_with_blank, blank).type(floatX)	# (batch, 2U+1)

    #print(sec_diag)

    ### RECURENCE RELATION SEEEMS TO BE VALID PATHS
    '''
        p.x for 3 tragets 2*U+1
     
     
        [1., 1., 0., 0., 0., 0., 0.],
        [0., 1., 1., 1., 0., 0., 0.],
        [0., 0., 1., 1., 0., 0., 0.],
        [0., 0., 0., 1., 1., 1., 0.],
        [0., 0., 0., 0., 1., 1., 0.],
        [0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 0., 1.]]])
        
        
    '''
    #print("m_eye {} \n ".format(m_eye(length,k=1)))
    recurrence_relation = (m_eye(length) + m_eye(length, k=1)).repeat(batch, 1, 1) + m_eye(length, k=2).repeat(batch, 1, 1) * sec_diag[:, None, :]	# (batch, 2U+1, 2U+1)
    #print("Recurence realation size {}".format(recurrence_relation.size()))
    #print(recurrence_relation)

    '''## with eps_nan zeros go to -1e8'''
    recurrence_relation = eps_nan * (T.ones_like(recurrence_relation) - recurrence_relation)




    #print("Recurence realation aftter size {}".format(recurrence_relation.size()))

    # alpha
    #print("pred size {}\n ones {}".format(pred[0, :, :2].size(),(T.ones(batch, 2*U-1).type(floatX)*eps_nan).size()))
    #print("pred[0,:,:2] {}\n ones {}".format(pred[0, :, :2],T.ones(batch, 2*U-1).type(floatX)*eps_nan))


    ''' 
        alpha_t is forward path probability 
        Initialize alpha_t  with pred[:2] alpha(0) , alpha(1) as in paper
        
    
    '''
    alpha_t = T.cat((pred[0, :, :2], T.ones(batch, 2*U-1).type(floatX)*eps_nan), dim=1) # (batch, 2U+1)
    #print(" Alpha {}".format(alpha_t))
    probability = alpha_t[None] # (1, batch, 2U+1)

    # dynamic programming
    # (T, batch, 2U+1)
    for t in T.arange(1,Time).type(longX):
        print(alpha_t.size())
        #print(" sun in for loop sizes apha_t{} pred[t]{} log_bath{}".format(alpha_t.size(),pred[t].size(),log_batch_dot(alpha_t, recurrence_relation).size()))
        alpha_t = pred[t]
        probability = T.cat((probability,alpha_t[None]), dim=0)
        print(probability.size())
    print("probability size {}".format(probability.size()))
    """
        pred_len-1= T-1
    """
    labels_2 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len-1]
    labels_1 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len]
    labels_prob = log_sum_exp(labels_2, labels_1)
    #pdb.set_trace()

    cost = -labels_prob
    return cost



















def ctc_cost(out, targets, sizes, target_sizes):
#    A batched version for uni_alpha_cost
#    param out: (Time, batch, voca_size+1)
#    param targets: targets without splited
#    param sizes: size for out (N)
#    param target_sizes: size for targets (N)

    Time = out.size(0)
    pred = T.nn.functional.log_softmax(out, dim=-1)

    offset = 0
    batch = target_sizes.size(0)
    target_max = target_sizes.max().item()
    target = T.zeros(batch, target_max).type(longX)

    for index, (target_size, size) in enumerate(zip(target_sizes, sizes)):
        target[index, :target_size.item()] = targets[offset: offset+target_size.item()].data
        offset += target_size.item()

    if not cuda:
        costs = ctc_loss_log(pred.cpu(), sizes.data.type(longX), target, target_sizes.data.type(longX))
    else:
        costs = ctc_loss_log(pred, sizes.data.type(longX), target, target_sizes.data.type(longX))
    return costs.sum()

def test_seg_ctc(use_mine=True):
    # (T, voca_size+1)
    pred_np = np.array([[0.5, 0.4, 0.1], [0.3, 0.1, 0.6], [0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])[:, None]
    pred_np = np.log(np.tile(pred_np, (1,2,1)))
    # (U)
    token_np = np.array([2, 2, 1, 2])

    pred = Variable(floatX(pred_np), requires_grad=True)
    token = Variable(T.IntTensor(token_np))
    sizes = Variable(T.IntTensor(np.array([4, 4])))
    target_sizes = Variable(T.IntTensor(np.array([2, 2])))





    size = 50
    voca_size = 10
    n = 1

    np.random.seed(1234)
    pred_len_np = np.ones([n])*size
    pred_np = np.random.random([size, n, voca_size+1])
    pred_np = np.log(pred_np)

    token_len_np = np.random.randint(low=9, high=10, size=n)
    token_np = np.random.randint(voca_size, size=token_len_np.sum())+1

    pred = Variable(floatX(pred_np), requires_grad=True)
    token = Variable(T.IntTensor(token_np))
    sizes = Variable(T.IntTensor(pred_len_np))
    target_sizes = Variable(T.IntTensor(token_len_np))

    print("{} {} {} {} ".format(pred.size(), token.size(), sizes.size(), target_sizes.size()))

    for i in range(400):

        cost = ctc_cost(pred, token, sizes, target_sizes)
        print("M-ctc ",cost.data.item())


        criterion = T.nn.CTCLoss(blank=0,reduction='sum')
        #cost = criterion(pred, token, sizes, target_sizes)
        #print(" CTC CPU ",cost.data.item())

        optimizer = T.optim.SGD([pred], lr=3e-1, momentum=0.9, nesterov=True)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

if __name__ == '__main__':
    print('_________')
    test_seg_ctc(use_mine=True)









def ctc_loss(pred, pred_len, token, token_len, blank=0):
    '''
    :param pred: (Time, batch, voca_size+1)
    :param pred_len: (batch)
    :param token: (batch, U)
    :param token_len: (batch)
    '''
    Time, batch = pred.size(0), pred.size(1)
    U = token.size(1)
    eps = 0

    # token_with_blank
    token_with_blank = T.cat((T.zeros(batch, U, 1).type(longX), token[:, :, None]), dim=2).view(batch, -1)    # (batch, 2U)
    token_with_blank = T.cat((token_with_blank, T.zeros(batch, 1).type(longX)), dim=1)  # (batch, 2U+1)
    length = token_with_blank.size(1)

    pred = pred[T.arange(0, Time).type(longX)[:, None, None], T.arange(0, batch).type(longX)[None, :, None], token_with_blank[None, :]]  # (T, batch, 2U+1)

    # recurrence relation
    sec_diag = T.cat((T.zeros((batch, 2)).type(floatX), T.ne(token_with_blank[:, :-2], token_with_blank[:, 2:]).type(floatX)), dim=1) * T.ne(token_with_blank, blank).type(floatX)	# (batch, 2U+1)
    recurrence_relation = (m_eye(length) + m_eye(length, k=1)).repeat(batch, 1, 1) + m_eye(length, k=2).repeat(batch, 1, 1) * sec_diag[:, None, :]	# (batch, 2U+1, 2U+1)

    # alpha
    alpha_t = T.cat((pred[0, :, :2], T.zeros(batch, 2*U-1).type(floatX)), dim=1) # (batch, 2U+1)
    probability = alpha_t[None] # (1, batch, 2U+1)

    # dynamic programming
    # (T, batch, 2U+1)
    for t in T.arange(1, Time).type(longX):
        alpha_t = T.bmm(alpha_t[:, None], recurrence_relation)[:, 0] * pred[t]
        probability = T.cat((probability, alpha_t[None]), dim=0)

    labels_2 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len-1]
    labels_1 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len]
    labels_prob = labels_2 + labels_1

    cost = -T.log(labels_prob+eps)
    return cost
