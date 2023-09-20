import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import wer_generic


class CTCPrefixScore():
    '''
    CTC Prefix score calculator
    An implementation of Algo. 2 in https://www.merl.com/publications/docs/TR2017-190.pdf (Watanabe et. al.)
    Reference (official implementation): https://github.com/espnet/espnet/tree/master/espnet/nets
    '''

    def __init__(self, x):
        self.logzero = -100000000.0
        self.blank = 0
        self.eos = 1
        self.x = x.cpu().numpy()[0]
        self.odim = x.shape[-1]
        self.input_length = len(self.x)

    def init_state(self):
        # 0 = non-blank, 1 = blank
        r = np.full((self.input_length, 2), self.logzero, dtype=np.float32)

        # Accumalate blank at each step
        r[0, 1] = self.x[0, self.blank]
        for i in range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        return r

    def full_compute(self, g, r_prev):
        '''Given prefix g, return the probability of all possible sequence y (where y = concat(g,c))
           This function computes all possible tokens for c (memory inefficient)'''
        prefix_length = len(g)
        last_char = g[-1] if prefix_length > 0 else 0

        # init. r
        r = np.full((self.input_length, 2, self.odim),
                    self.logzero, dtype=np.float32)

        # start from len(g) because is impossible for CTC to generate |y|>|X|
        start = max(1, prefix_length)

        if prefix_length == 0:
            r[0, 0, :] = self.x[0, :]  # if g = <sos>

        psi = r[start - 1, 0, :]

        phi = np.logaddexp(r_prev[:, 0], r_prev[:, 1])

        for t in range(start, self.input_length):
            # prev_blank
            prev_blank = np.full((self.odim), r_prev[t - 1, 1], dtype=np.float32)
            # prev_nonblank
            prev_nonblank = np.full(
                (self.odim), r_prev[t - 1, 0], dtype=np.float32)
            prev_nonblank[last_char] = self.logzero

            phi = np.logaddexp(prev_nonblank, prev_blank)
            # P(h|current step is non-blank) = [ P(prev. step = y) + P()]*P(c)
            r[t, 0, :] = np.logaddexp(r[t - 1, 0, :], phi) + self.x[t, :]
            # P(h|current step is blank) = [P(prev. step is blank) + P(prev. step is non-blank)]*P(now=blank)
            r[t, 1, :] = np.logaddexp(
                r[t - 1, 1, :], r[t - 1, 0, :]) + self.x[t, self.blank]
            psi = np.logaddexp(psi, phi + self.x[t, :])

        # psi[self.eos] = np.logaddexp(r_prev[-1,0], r_prev[-1,1])
        return psi, np.rollaxis(r, 2)

    def cheap_compute(self, g, r_prev, candidates):
        '''Given prefix g, return the probability of all possible sequence y (where y = concat(g,c))
           This function considers only those tokens in candidates for c (memory efficient)'''
        prefix_length = len(g)
        odim = len(candidates)
        last_char = g[-1] if prefix_length > 0 else 0

        # init. r
        r = np.full((self.input_length, 2, len(candidates)),
                    self.logzero, dtype=np.float32)

        # start from len(g) because is impossible for CTC to generate |y|>|X|
        start = max(1, prefix_length)

        if prefix_length == 0:
            r[0, 0, :] = self.x[0, candidates]  # if g = <sos>

        psi = r[start - 1, 0, :]
        # Phi = (prev_nonblank,prev_blank)
        sum_prev = np.logaddexp(r_prev[:, 0], r_prev[:, 1])
        phi = np.repeat(sum_prev[..., None], odim, axis=-1)
        # Handle edge case : last tok of prefix in candidates
        if prefix_length > 0 and last_char in candidates:
            phi[:, candidates.index(last_char)] = r_prev[:, 1]

        for t in range(start, self.input_length):
            # prev_blank
            # prev_blank = np.full((odim), r_prev[t-1, 1], dtype=np.float32)
            # prev_nonblank
            # prev_nonblank = np.full((odim), r_prev[t-1, 0], dtype=np.float32)
            # phi = np.logaddexp(prev_nonblank, prev_blank)
            # P(h|current step is non-blank) =  P(prev. step = y)*P(c)
            r[t, 0, :] = np.logaddexp(r[t - 1, 0, :], phi[t - 1]) + self.x[t, candidates]
            # P(h|current step is blank) = [P(prev. step is blank) + P(prev. step is non-blank)]*P(now=blank)
            r[t, 1, :] = np.logaddexp(r[t - 1, 1, :], r[t - 1, 0, :]) + self.x[t, self.blank]
            psi = np.logaddexp(psi, phi[t - 1,] + self.x[t, candidates])

        # P(end of sentence) = P(g)
        if self.eos in candidates:
            psi[candidates.index(self.eos)] = sum_prev[-1]
        return psi, np.rollaxis(r, 2)


CTC_BEAM_RATIO = 1.5  # DO NOT CHANGE THIS, MAY CAUSE OOM
LOG_ZERO = -10000000.0  # Log-zero for CTC


class BeamDecoder(nn.Module):
    ''' Beam decoder for ASR '''

    def __init__(self, asr, emb_decoder, beam_size, min_len_ratio, max_len_ratio,
                 lm_path='', lm_config='', lm_weight=0.0, ctc_weight=0.0):
        super().__init__()
        # Setup
        self.beam_size = beam_size
        self.min_len_ratio = min_len_ratio
        self.max_len_ratio = max_len_ratio
        self.asr = asr

        # ToDo : implement pure ctc decode
        assert self.asr.enable_att

        # Additional decoding modules
        self.apply_ctc = ctc_weight > 0
        if self.apply_ctc:
            assert self.asr.ctc_weight > 0, 'ASR was not trained with CTC decoder'
            self.ctc_w = ctc_weight
            self.ctc_beam_size = int(CTC_BEAM_RATIO * self.beam_size)

        self.apply_lm = lm_weight > 0
        if self.apply_lm:
            self.lm_w = lm_weight
            self.lm_path = lm_path

            self.lm = RNNLM(self.asr.vocab_size, **lm_config['model'])
            self.lm.load_state_dict(torch.load(
                self.lm_path, map_location='cpu')['model'])
            self.lm.eval()

        self.apply_emb = emb_decoder is not None
        if self.apply_emb:
            self.emb_decoder = emb_decoder

    def create_msg(self):
        msg = ['Decode spec| Beam size = {}\t| Min/Max len ratio = {}/{}'.format(
            self.beam_size, self.min_len_ratio, self.max_len_ratio)]
        if self.apply_ctc:
            msg.append(
                '           |Joint CTC decoding enabled \t| weight = {:.2f}\t'.format(self.ctc_w))
        if self.apply_lm:
            msg.append('           |Joint LM decoding enabled \t| weight = {:.2f}\t| src = {}'.format(
                self.lm_w, self.lm_path))
        if self.apply_emb:
            msg.append('           |Joint Emb. decoding enabled \t| weight = {:.2f}'.format(
                self.lm_w, self.emb_decoder.fuse_lambda.mean().cpu().item()))

        return msg

    def forward(self, audio_feature, feature_len):
        # Init.
        assert audio_feature.shape[0] == 1, "Batchsize == 1 is required for beam search"
        batch_size = audio_feature.shape[0]
        device = audio_feature.device
        dec_state = self.asr.decoder.init_state(
            batch_size)  # Init zero states
        self.asr.attention.reset_mem()  # Flush attention mem
        # Max output len set w/ hyper param.
        max_output_len = int(
            np.ceil(feature_len.cpu().item() * self.max_len_ratio))
        # Min output len set w/ hyper param.
        min_output_len = int(
            np.ceil(feature_len.cpu().item() * self.min_len_ratio))
        # Store attention map if location-aware
        store_att = self.asr.attention.mode == 'loc'
        prev_token = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=device)  # Start w/ <sos>
        # Cache of beam search
        final_hypothesis, next_top_hypothesis = [], []
        # Incase ctc is disabled
        ctc_state, ctc_prob, candidates, lm_state = None, None, None, None

        # Encode
        encode_feature, encode_len = self.asr.encoder(
            audio_feature, feature_len)

        # CTC decoding
        if self.apply_ctc:
            ctc_output = F.log_softmax(
                self.asr.ctc_layer(encode_feature), dim=-1)
            ctc_prefix = CTCPrefixScore(ctc_output)
            ctc_state = ctc_prefix.init_state()

        # Start w/ empty hypothesis
        prev_top_hypothesis = [Hypothesis(decoder_state=dec_state, output_seq=[],
                                          output_scores=[], lm_state=None, ctc_prob=0,
                                          ctc_state=ctc_state, att_map=None)]
        # Attention decoding
        for t in range(max_output_len):
            for hypothesis in prev_top_hypothesis:
                # Resume previous step
                prev_token, prev_dec_state, prev_attn, prev_lm_state, prev_ctc_state = hypothesis.get_state(
                    device)
                self.asr.set_state(prev_dec_state, prev_attn)

                # Normal asr forward
                attn, context = self.asr.attention(
                    self.asr.decoder.get_query(), encode_feature, encode_len)
                asr_prev_token = self.asr.pre_embed(prev_token)
                decoder_input = torch.cat([asr_prev_token, context], dim=-1)
                cur_prob, d_state = self.asr.decoder(decoder_input)

                # Embedding fusion (output shape 1xV)
                if self.apply_emb:
                    _, cur_prob = self.emb_decoder(d_state, cur_prob, return_loss=False)
                else:
                    cur_prob = F.log_softmax(cur_prob, dim=-1)

                # Perform CTC prefix scoring on limited candidates (else OOM easily)
                if self.apply_ctc:
                    # TODO : Check the performance drop for computing part of candidates only
                    _, ctc_candidates = cur_prob.squeeze(0).topk(self.ctc_beam_size, dim=-1)
                    candidates = ctc_candidates.cpu().tolist()
                    ctc_prob, ctc_state = ctc_prefix.cheap_compute(
                        hypothesis.outIndex, prev_ctc_state, candidates)
                    # TODO : study why ctc_char (slightly) > 0 sometimes
                    ctc_char = torch.FloatTensor(ctc_prob - hypothesis.ctc_prob).to(device)

                    # Combine CTC score and Attention score (HACK: focus on candidates, block others)
                    hack_ctc_char = torch.zeros_like(cur_prob).data.fill_(LOG_ZERO)
                    for idx, char in enumerate(candidates):
                        hack_ctc_char[0, char] = ctc_char[idx]
                    cur_prob = (1 - self.ctc_w) * cur_prob + self.ctc_w * hack_ctc_char  # ctc_char
                    cur_prob[0, 0] = LOG_ZERO  # Hack to ignore <sos>

                # Joint RNN-LM decoding
                if self.apply_lm:
                    # assuming batch size always 1, resulting 1x1
                    lm_input = prev_token.unsqueeze(1)
                    lm_output, lm_state = self.lm(
                        lm_input, torch.ones([batch_size]), hidden=prev_lm_state)
                    # assuming batch size always 1,  resulting 1xV
                    lm_output = lm_output.squeeze(0)
                    cur_prob += self.lm_w * lm_output.log_softmax(dim=-1)

                # Beam search
                # Note: Ignored batch dim.
                topv, topi = cur_prob.squeeze(0).topk(self.beam_size)
                prev_attn = self.asr.attention.att_layer.prev_att.cpu() if store_att else None
                final, top = hypothesis.addTopk(topi, topv, self.asr.decoder.get_state(), att_map=prev_attn,
                                                lm_state=lm_state, ctc_state=ctc_state, ctc_prob=ctc_prob,
                                                ctc_candidates=candidates)
                # Move complete hyps. out
                if final is not None and (t >= min_output_len):
                    final_hypothesis.append(final)
                    if self.beam_size == 1:
                        return final_hypothesis
                next_top_hypothesis.extend(top)

            # Sort for top N beams
            next_top_hypothesis.sort(key=lambda o: o.avgScore(), reverse=True)
            prev_top_hypothesis = next_top_hypothesis[:self.beam_size]
            next_top_hypothesis = []

        # Rescore all hyp (finished/unfinished)
        final_hypothesis += prev_top_hypothesis
        final_hypothesis.sort(key=lambda o: o.avgScore(), reverse=True)

        return final_hypothesis[:self.beam_size]


class Hypothesis:
    '''Hypothesis for beam search decoding.
       Stores the history of label sequence & score
       Stores the previous decoder state, ctc state, ctc score, lm state and attention map (if necessary)'''

    def __init__(self, decoder_state, output_seq, output_scores, lm_state, ctc_state, ctc_prob, att_map):
        assert len(output_seq) == len(output_scores)
        # attention decoder
        self.decoder_state = decoder_state
        self.att_map = att_map

        # RNN language model
        if type(lm_state) is tuple:
            self.lm_state = (lm_state[0].cpu(),
                             lm_state[1].cpu())  # LSTM state
        elif lm_state is None:
            self.lm_state = None  # Init state
        else:
            self.lm_state = lm_state.cpu()  # GRU state

        # Previous outputs
        self.output_seq = output_seq  # Prefix, List of list
        self.output_scores = output_scores  # Prefix score, list of float

        # CTC decoding
        self.ctc_state = ctc_state  # List of np
        self.ctc_prob = ctc_prob  # List of float

    def avgScore(self):
        '''Return the averaged log probability of hypothesis'''
        assert len(self.output_scores) != 0
        return sum(self.output_scores) / len(self.output_scores)

    def addTopk(self, topi, topv, decoder_state, att_map=None,
                lm_state=None, ctc_state=None, ctc_prob=0.0, ctc_candidates=[]):
        '''Expand current hypothesis with a given beam size'''
        new_hypothesis = []
        term_score = None
        ctc_s, ctc_p = None, None
        beam_size = topi.shape[-1]

        for i in range(beam_size):
            # Detect <eos>
            if topi[i].item() == 1:
                term_score = topv[i].cpu()
                continue

            idxes = self.output_seq[:]  # pass by value
            scores = self.output_scores[:]  # pass by value
            idxes.append(topi[i].cpu())
            scores.append(topv[i].cpu())
            if ctc_state is not None:
                # ToDo: Handle out-of-candidate case.
                idx = ctc_candidates.index(topi[i].item())
                ctc_s = ctc_state[idx, :, :]
                ctc_p = ctc_prob[idx]
            new_hypothesis.append(Hypothesis(decoder_state,
                                             output_seq=idxes, output_scores=scores, lm_state=lm_state,
                                             ctc_state=ctc_s, ctc_prob=ctc_p, att_map=att_map))
        if term_score is not None:
            self.output_seq.append(torch.tensor(1))
            self.output_scores.append(term_score)
            return self, new_hypothesis
        return None, new_hypothesis

    def get_state(self, device):
        prev_token = self.output_seq[-1] if len(self.output_seq) != 0 else 0
        prev_token = torch.LongTensor([prev_token]).to(device)
        att_map = self.att_map.to(device) if self.att_map is not None else None
        if type(self.lm_state) is tuple:
            lm_state = (self.lm_state[0].to(device),
                        self.lm_state[1].to(device))  # LSTM state
        elif self.lm_state is None:
            lm_state = None  # Init state
        else:
            lm_state = self.lm_state.to(
                device)  # GRU state
        return prev_token, self.decoder_state, att_map, lm_state, self.ctc_state

    @property
    def outIndex(self):
        return [i.item() for i in self.output_seq]


class RNNLM(nn.Module):
    ''' RNN Language Model '''

    def __init__(self, vocab_size, emb_tying, emb_dim, module, dim, n_layers, dropout):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.emb_tying = emb_tying
        if emb_tying:
            assert emb_dim == dim, "Output dim of RNN should be identical to embedding if using weight tying."
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.rnn = getattr(nn, module.upper())(
            emb_dim, dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        if not self.emb_tying:
            self.trans = nn.Linear(emb_dim, vocab_size)

    def create_msg(self):
        # Messages for user
        msg = ['Model spec.| RNNLM weight tying = {}, # of layers = {}, dim = {}'.format(
            self.emb_tying, self.n_layers, self.dim)]
        return msg

    def forward(self, x, lens, hidden=None):
        emb_x = self.dp1(self.emb(x))
        if not self.training:
            self.rnn.flatten_parameters()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb_x, lens, batch_first=True, enforce_sorted=False)
        # output: (seq_len, batch, hidden)
        outputs, hidden = self.rnn(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        if self.emb_tying:
            outputs = F.linear(self.dp2(outputs), self.emb.weight)
        else:
            outputs = self.trans(self.dp2(outputs))
        return outputs, hidden


"""
Author: Awni Hannun

This is an example CTC decoder written in Python. The code is
intended to be a simple example and is not designed to be
especially efficient.

The algorithm is a prefix beam search for a model trained
with the CTC loss function.

For more details checkout either of these references:
  https://distill.pub/2017/ctc/#inference
  https://arxiv.org/abs/1408.2873

"""

import numpy as np
import math
import collections

NEG_INF = -float("inf")


def make_new_beam():
    fn = lambda: (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)


def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max)
                       for a in args))
    return a_max + lsp


def beam_search_decode(probs, target, id2w, beam_size=10, blank=0):
    """
    Performs inference for the given output probabilities.

    Arguments:
      probs: The output probabilities (e.g. log post-softmax) for each
        time step. Should be an array of shape (time x output dim).
      beam_size (int): Size of the beam to use during inference.
      blank (int): Index of the CTC blank label.

    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    T, S = probs.shape

    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    beam = [(tuple(), (0.0, NEG_INF))]

    for t in range(T):  # Loop over time

        # A default dictionary to store the next step candidates.
        next_beam = make_new_beam()

        for s in range(S):  # Loop over vocab
            p = probs[t, s]

            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (p_b, p_nb) in beam:  # Loop over beam

                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == blank:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s != end_t:
                    n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                    # We don't include the previous probability of not ending
                    # in blank (p_nb) if s is repeated at the end. The CTC
                    # algorithm merges characters not separated by a blank.
                    n_p_nb = logsumexp(n_p_nb, p_b + p)

                # *NB* this would be a good place to include an LM score.
                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if s == end_t:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = logsumexp(n_p_nb, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                      key=lambda x: logsumexp(*x[1]),
                      reverse=True)
        beam = beam[:beam_size]

    ref = ''
    refs = target.squeeze().cpu().numpy()
    if (target.size(1) == 1):
        # print('ekmek shape ',refs.shape,' ',refs)
        ref += id2w[int(refs)] + ' '
    else:
        for i in range(target.size(1)):
            ref += id2w[refs[i]] + ' '

    decoded_sentence = ''
    best_beam_preds = beam[0][0]
    best = beam[0]
    # print(beam)
    # print(best_beam_preds)
    prev_found_word = ''

    for i in range(len(best_beam_preds)):
        temp = id2w[best_beam_preds[i]]
        if (temp != prev_found_word):
            decoded_sentence += temp + ' '
        prev_found_word = temp

    temp_wer, C, S, I, D = wer_generic(ref, decoded_sentence)
    return best[0], -logsumexp(*best[1]), decoded_sentence, temp_wer
