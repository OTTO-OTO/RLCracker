from argparse import Namespace
import os
import subprocess
import numpy as np
from sacremoses import MosesTokenizer
from metrics.p_sp_utils.models import load_model
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from metrics.p_sp_utils.data_utils import get_df
from metrics.p_sp_utils.evaluate_sts import Example


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

class FileSim(object):

    def __init__(self):
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

    def score(self, params, batcher, input1, input2, use_sent_transformers=False, use_cuda=False, sentence_model=None):
        sys_scores = []
        if not use_sent_transformers:
            for ii in range(0, len(input1), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)

                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)
        else:
            #Compute embedding for both lists
            device = 'cuda' if use_cuda else 'cpu'
            for i in range(len(input1)):
                if sentence_model is None:
                    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
                    # sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device) # 'cuda:1' cpu
                embedding_1= sentence_model.encode(input1[i], convert_to_tensor=True)
                embedding_2 = sentence_model.encode(input2[i], convert_to_tensor=True)

                score = util.pytorch_cos_sim(embedding_1, embedding_2)
                sys_scores.append(score.item())
        return sys_scores


def batcher(params, batch):
    new_batch = []
    for p in batch:
        if params.tokenize:
            tok = params.entok.tokenize(p, escape=False)
            p = " ".join(tok)
        if params.lower_case:
            p = p.lower()
        p = params.sp.EncodeAsPieces(p)
        p = " ".join(p)
        p = Example(p, params.lower_case)
        p.populate_embeddings(params.model.vocab, params.model.zero_unk, params.model.ngrams)
        new_batch.append(p)
    x, l = params.model.torchify_batch(new_batch)
    vecs = params.model.encode(x, l)
    return vecs.detach().cpu().numpy()


def GRPO_P_SP(similarity_model, psp_args, input1, input2, use_sent_transformers=False):
    s = FileSim()
    similarity_model.eval()
    entok = MosesTokenizer(lang='en')
    new_psp_args = Namespace(batch_size=32, entok=entok, sp=similarity_model.sp,
                    params=psp_args, model=similarity_model, lower_case=similarity_model.args.lower_case,
                    tokenize=similarity_model.args.tokenize)
    
    scores = s.score(new_psp_args, batcher, input1, input2, use_sent_transformers)
    return scores


def GRPO_Sim(similarity_model, input1, input2):
    s = FileSim()
    scores = s.score(None, None, input1, input2, True, use_cuda=True, sentence_model=similarity_model)
    return scores


def cal_p_sp(input1, input2, use_sent_transformers=False, use_cuda=False):
    s = FileSim()
    if use_sent_transformers:
        scores = s.score(None, None, input1, input2, use_sent_transformers, use_cuda)
    else:
        args = {
                'gpu': 1 if use_cuda else 0, #1 if torch.cuda.is_available() else 0,
                'load_file': 'metrics/p_sp_utils/psp/model.para.lc.100.pt',
                'sp_model': 'metrics/p_sp_utils/psp/paranmt.model',
                'gpu_id':0
            }
        model, _ = load_model(None, args)
        model.eval()

        entok = MosesTokenizer(lang='en')
        new_args = Namespace(batch_size=32, entok=entok, sp=model.sp,
                        params=args, model=model, lower_case=model.args.lower_case,
                        tokenize=model.args.tokenize)
        
        scores = s.score(new_args, batcher, input1, input2, use_sent_transformers, use_cuda)

    return scores