""" batching """
import random
from collections import defaultdict

from toolz.sandbox import unzip
from cytoolz import curry, concat, compose
from cytoolz import curried

import torch
import torch.multiprocessing as mp
from pytorch_pretrained_bert.tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    '/path/to/uncased_L-24_H-1024_A-16/vocab.txt')
MAX_ARTICLE_LEN = 512

# Batching functions
def coll_fn_extract(data):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts = d
        return source_sents and extracts
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch

@curry
def tokenize(max_len, emb_type, texts):
    if emb_type == 'W2V':
        return [t.lower().split()[:max_len] for t in texts]
    else:
        truncated_article = []
        left = MAX_ARTICLE_LEN - 2
        for sentence in texts:
            tokens = tokenizer.tokenize(sentence)
            tokens = tokens[:max_len]
            if left >= len(tokens):
                truncated_article.append(tokens)
                left -= len(tokens)
            else:
                truncated_article.append(tokens[0:left])
                break
        return truncated_article

def conver2id(unk, word2id, words_list):
    word2id = defaultdict(lambda: unk, word2id)
    return [[word2id[w] for w in words] for words in words_list]

@curry
def prepro_fn_extract(max_src_len, max_src_num, emb_type, batch):
    def prepro_one(sample):
        source_sents, extracts = sample
        tokenized_sents = tokenize(max_src_len, emb_type, source_sents)[:max_src_num]
        cleaned_extracts = list(filter(lambda e: e < len(tokenized_sents),
                                       extracts))
        return tokenized_sents, cleaned_extracts
    batch = list(map(prepro_one, batch))
    return batch

@curry
def convert_batch_extract_ptr(unk, word2id, emb_type, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        if emb_type == 'W2V':
            id_sents = conver2id(unk, word2id, source_sents)
        else:
            id_sents = [tokenizer.convert_tokens_to_ids(sentence) 
                        for sentence in source_sents]
        return id_sents, extracts
    batch = list(map(convert_one, batch))
    return batch

@curry
def pad_batch_tensorize(inputs, pad, cuda=True):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    max_len = max(len(ids) for ids in inputs)
    tensor_shape = (batch_size, max_len)
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        tensor[i, :len(ids)] = tensor_type(ids)
    return tensor

@curry
def batchify_fn_extract_ptr(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))

    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )
    
    fw_args = (sources, src_nums, tar_in)
    loss_args = (target, src_nums)
    return fw_args, loss_args

def _batch2q(loader, prepro, q, single_run=True):
    epoch = 0
    while True:
        for batch in loader:
            q.put(prepro(batch))
        if single_run:
            break
        epoch += 1
        q.put(epoch)
    q.put(None)

class BucketedGenerater(object):
    def __init__(self, loader, prepro,
                 sort_key, batchify,
                 single_run=True, queue_size=8, fork=True):
        self._loader = loader
        self._prepro = prepro
        self._sort_key = sort_key
        self._batchify = batchify
        self._single_run = single_run
        if fork:
            ctx = mp.get_context('forkserver')
            self._queue = ctx.Queue(queue_size)
        else:
            # for easier debugging
            self._queue = None
        self._process = None

    def __call__(self, batch_size: int):
        def get_batches(hyper_batch):
            indexes = list(range(0, len(hyper_batch), batch_size))
            if not self._single_run:
                # random shuffle for training batches
                random.shuffle(hyper_batch)
                random.shuffle(indexes)
            hyper_batch.sort(key=self._sort_key)
            for i in indexes:
                batch = self._batchify(hyper_batch[i:i+batch_size])
                yield batch

        if self._queue is not None:
            ctx = mp.get_context('forkserver')
            self._process = ctx.Process(
                target=_batch2q,
                args=(self._loader, self._prepro,
                      self._queue, self._single_run)
            )
            self._process.start()
            while True:
                d = self._queue.get()
                if d is None:
                    break
                if isinstance(d, int):
                    print('\nepoch {} done'.format(d))
                    continue
                yield from get_batches(d)
            self._process.join()
        else:
            i = 0
            while True:
                for batch in self._loader:
                    yield from get_batches(self._prepro(batch))
                if self._single_run:
                    break
                i += 1
                print('\nepoch {} done'.format(i))

    def terminate(self):
        if self._process is not None:
            self._process.terminate()
            self._process.join()
