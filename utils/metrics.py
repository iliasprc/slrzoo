import torch

# from jiwer import wer

DEL_PENALTY = 1
INS_PENALTY = 1
SUB_PENALTY = 1


def ctc_tensors_decode(output, target):
    probs = torch.nn.functional.softmax(output, dim=2)
    pred = probs.argmax(dim=2).squeeze(1)

    no_blanks_probs = probs[pred != 0]
    if no_blanks_probs.nelement() == 0:
        no_blanks_probs = probs[-1, :, :].unsqueeze(0)

    new_pred = no_blanks_probs.argmax(dim=2).squeeze(1)

    decoded_labels = no_blanks_probs[0].unsqueeze(0)
    prev_found_word = new_pred[0].item()

    for i in range(1, no_blanks_probs.size(0)):
        if new_pred[i].item() != prev_found_word:
            decoded_labels = torch.cat((decoded_labels, no_blanks_probs[i, :, :].unsqueeze(0)), dim=0)
        prev_found_word = new_pred[i].item()

    return decoded_labels.squeeze(1)



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracyv1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def generate_Discriminator_inputs(pred):
    # print(pred,pred[pred!=0])
    # remove blanks
    pred = pred[pred != 0]
    prev_found_word = ''
    prev = 0
    decoded_labels = []
    for i in range(pred.size(0)):
        if (pred[i] != prev):
            decoded_labels.append(pred[i])
        prev = pred[i]
    y_ = torch.stack(decoded_labels).unsqueeze(0)
    # print(decoded_labels, y_ , y_.shape)

    # for i in range(seq_len):
    #     # current label
    #     temp = id2w[pred[i]]
    #
    #     if (temp != 'blank'):
    #         # remove blanks
    #         if (prev != temp):
    #             # remove repeated labels
    #             if (temp != prev_found_word):
    #                 # check if previous word equals new word
    #
    #                 decoded_labels += temp + sep
    #             prev_found_word = temp
    #     prev = temp
    return y_


def label_per_output(id2w, pred, seq_len, sep=' '):
    decoded_labels = ''

    for i in range(seq_len):
        # current label
        temp = id2w[pred[i]]

        decoded_labels += temp + sep

    return decoded_labels


def wer_generic(ref, hyp, debug=False):
    r = ref.split()
    h = hyp.split()
    # print(r,h)
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB\t" + r[i] + "\t" + h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL\t" + r[i] + "\t" + "****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    # return (numSub + numDel + numIns) / (float) (len(r))
    wer_result = (numSub + numDel + numIns) / (float)(len(r))

    return wer_result, numCor, numSub, numIns, numDel


def word_error_rate_generic(output, target, id2w):
    probs = torch.nn.functional.softmax(output, dim=2)
    pred = probs.argmax(dim=2, keepdim=True).squeeze(1).squeeze(1).detach().cpu().numpy()
    ref = ''

    refs = target.squeeze().cpu().numpy()

    if (target.size(1) == 1):
        # print('ekmek shape ',refs.shape,' ',refs)
        ref += str(id2w[int(refs)]) + ' '
    else:
        for i in range(target.size(1)):
            ref += str(id2w[int(refs[i])]) + ' '


    sentence = greedy_decode(id2w, pred, output.size(0), ' ')
    temp_wer, C, S, I, D = wer_generic(ref, sentence)
    temp_wer = min(1, temp_wer)

    return temp_wer, sentence, C, S, I, D


def wer_alignment(output, target, id2w):
    probs = torch.nn.functional.softmax(output, dim=2)
    pred = probs.argmax(dim=2, keepdim=True).squeeze(1).squeeze(1).cpu().numpy()
    ref = ''
    refs = target.squeeze().cpu().numpy()
    if (target.size(1) == 1):

        ref += id2w[int(refs)] + ' '
    else:
        for i in range(target.size(1)):
            ref += id2w[refs[i]] + ' '

    sentence = ''
    sentence = greedy_decode(id2w, pred, output.size(0), ' ')
    label_of_out = label_per_output(id2w, pred, output.size(0), ',')
    temp_wer, C, S, I, D = wer_generic(ref, sentence)
    temp_wer = min(1, temp_wer)
    # print("REF : {} -> HYP : {} WER = {} ".format(ref, sentence, temp_wer))
    return label_of_out, temp_wer, sentence, C, S, I, D


def old_word_error_rate(output, target, id2w):
    ### requires jiwer
    probs = torch.nn.functional.softmax(output, dim=2)
    pred = probs.argmax(dim=2, keepdim=True).squeeze(1).squeeze(1).cpu().numpy()
    ref = ''
    refs = target.squeeze().cpu().numpy()
    if (target.size(1) == 1):
        ref += id2w[int(refs)] + ' '
    for i in range(target.size(1)):
        ref += id2w[refs[i]] + ' '

    s = ''
    s = greedy_decode(id2w, pred, output.size(0), ' ')
    temp_wer = min(1, wer(ref, s))

    return temp_wer, s


def accuracyv1(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def binary_accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    # print("Pred {:2f} , Target {:2f}".format(output.item(),target))
    # print(output.shape,target.shape)

    output = torch.sigmoid(output.view(-1))
    pred = output > 0.5
    truth = target > 0.5
    # print("Pred {} , Target {}".format(pred.item(),truth.item()))
    acc = pred.eq(truth).sum() / float(output.numel())
    return acc


def greedy_decode(id2w, pred, seq_len, sep=' '):
    prev_found_word = ''
    prev = ''
    decoded_labels = ''

    for i in range(seq_len):
        # current label
        temp = str(id2w[pred[i]])

        if (temp != 'blank'):
            # remove blanks
            if (prev != temp):
                # remove repeated labels
                if (temp != prev_found_word):
                    # check if previous word equals new word

                    decoded_labels += temp + sep
                prev_found_word = temp
        prev = temp
    return decoded_labels
