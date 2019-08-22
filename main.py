import argparse
import json
import os
from os.path import join, exists
import pickle as pkl
import random #
from time import time
from datetime import timedelta

from cytoolz import compose

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.extract import Summarizer
from model.util import sequence_loss
from training import get_basic_grad_fn, basic_validate
from training import BasicPipeline, BasicTrainer
from decoding import Extractor, DecodeDataset
from decoding import sort_ckpt, get_n_ext
from evaluate import eval_rouge

from utils import PAD, UNK
from utils import make_vocab, make_embedding

from data.data import CnnDmDataset
from data.batcher import tokenize
from data.batcher import coll_fn_extract
from data.batcher import prepro_fn_extract
from data.batcher import convert_batch_extract_ptr
from data.batcher import batchify_fn_extract_ptr
from data.batcher import BucketedGenerater

BUCKET_SIZE = 6400

DATA_DIR = './CNNDM'

class ExtractDataset(CnnDmDataset):
    """ article sentences -> extraction indices
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, extracts = js_data['article'], js_data['label']
        return art_sents, extracts

def set_parameters(args):
    if args.encoder == 'BiLSTM':
        args.lr = 1e-4 if args.emb_type == 'W2V' else 5e-5
        args.clip = 2.0
        args.encoder_layer = 1 # Actually 2 layers (bidirectional)
        args.encoder_hidden = 512
    elif args.encoder == 'Transformer' and args.decoder == 'PN':
        args.encoder_layer = 4
        args.encoder_hidden = 512
    elif args.encoder == 'Transformer' and args.decoder == 'SL':
        args.lr = 5e-5
        args.decay = 0.8
        args.patience = 10
        args.encoder_layer = 12
        args.encoder_hidden = 512
    elif args.encoder == 'DeepLSTM' and args.decoder == 'PN':
        args.batch = 16
        args.ckpt_freq = 6000
        args.encoder_layer = 4
        args.encoder_hidden = 2048
    elif args.encoder == 'DeepLSTM' and args.decoder == 'SL':
        args.lr = 5e-5
        args.batch = 16
        args.ckpt_freq = 6000
        args.encoder_layer = 8
        args.encoder_hidden = 2048
    return args

def build_batchers(decoder, emb_type, word2id, cuda, debug):
    prepro = prepro_fn_extract(args.max_word, args.max_sent, emb_type)
    def sort_key(sample):
        src_sents, _ = sample
        return len(src_sents)
    batchify_fn = batchify_fn_extract_ptr
    convert_batch = convert_batch_extract_ptr
    batchify = compose(batchify_fn(PAD, cuda=cuda),
                       convert_batch(UNK, word2id, emb_type))

    train_loader = DataLoader(
        ExtractDataset('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)

    val_loader = DataLoader(
        ExtractDataset('val'), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)
    return train_batcher, val_batcher


def configure_net(encoder, decoder, emb_type, vocab_size, emb_dim, 
                  conv_hidden, encoder_hidden, encoder_layer):
    model_args = {}
    model_args['encoder']         = encoder
    model_args['decoder']         = decoder
    model_args['emb_type']        = emb_type
    model_args['vocab_size']      = vocab_size
    model_args['emb_dim']         = emb_dim
    model_args['conv_hidden']     = conv_hidden
    model_args['encoder_hidden']  = encoder_hidden
    model_args['encoder_layer']   = encoder_layer

    model = Summarizer(**model_args)
    return model, model_args


def configure_training(decoder, opt, lr, clip_grad, lr_decay, batch_size):
    """ supports Adam optimizer only"""
    assert opt in ['adam']

    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay

    ce = lambda logit, target: F.cross_entropy(logit, target, reduction='none')
    def criterion(logits, targets, sent_num, decoder):
        return sequence_loss(logits, targets, sent_num, decoder, ce, pad_idx=-1)

    return criterion, train_params


def train(args):

    assert args.encoder  in ['BiLSTM', 'DeepLSTM', 'Transformer']
    assert args.decoder  in ['SL', 'PN']
    assert args.emb_type in ['W2V', 'BERT']

    # create data batcher, vocabulary
    # batcher
    with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
        wc = pkl.load(f)
    word2id = make_vocab(wc, args.vsize)
    train_batcher, val_batcher = build_batchers(args.decoder, args.emb_type, 
                                                word2id, args.cuda, args.debug)

    # make model
    model, model_args = configure_net(args.encoder, args.decoder, args.emb_type, len(word2id), 
                                      args.emb_dim, args.conv_hidden, args.encoder_hidden, 
                                      args.encoder_layer)
    
    if args.emb_type == 'W2V':
        # NOTE: the pretrained embedding having the same dimension
        #       as args.emb_dim should already be trained
        w2v_path='./CNNDM/word2vec/word2vec.128d.226k.bin'
        embedding, _ = make_embedding(
            {i: w for w, i in word2id.items()}, w2v_path)
        model.set_embedding(embedding)

    # configure training setting
    criterion, train_params = configure_training(
        args.decoder, 'adam', args.lr, args.clip, args.decay, args.batch
    )

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)
    meta = {}
    meta['model_args']    = model_args
    meta['traing_params'] = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # prepare trainer
    val_fn = basic_validate(model, criterion, args.decoder)
    grad_fn = get_basic_grad_fn(model, args.clip)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=2e-5,
                                  patience=args.lr_p)

    if args.cuda:
        model = model.cuda()
    pipeline = BasicPipeline(model, args.decoder, 
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion, optimizer, grad_fn)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler)
    
    # for name, para in net.named_parameters():
    #     if para.requires_grad:
    #         print(name)

    print('Start training with the following hyper-parameters:')
    print(meta)
    trainer.train()

def test(args, split):
    ext_dir = args.path
    ckpts = sort_ckpt(ext_dir)
    
    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    dataset = DecodeDataset(split)

    n_data = len(dataset)
    loader = DataLoader(
             dataset, batch_size=args.batch, shuffle=False, num_workers=4,
             collate_fn=coll
    )

    # decode and evaluate top 5 models
    os.mkdir(join(args.path, 'decode'))
    os.mkdir(join(args.path, 'ROUGE'))
    for i in range(min(5, len(ckpts))):
        print('Start loading checkpoint {} !'.format(ckpts[i]))
        cur_ckpt = torch.load(
                   join(ext_dir, 'ckpt/{}'.format(ckpts[i]))
        )['state_dict']
        extractor = Extractor(ext_dir, cur_ckpt, args.emb_type, cuda=args.cuda)
        save_path = join(args.path, 'decode/{}'.format(ckpts[i]))
        os.mkdir(save_path)

        # decoding
        ext_list = []
        cur_idx = 0
        start = time()
        with torch.no_grad():
            for raw_article_batch in loader:
                tokenized_article_batch = map(tokenize(None, args.emb_type), raw_article_batch)
                for raw_art_sents in tokenized_article_batch:
                    ext_idx = extractor(raw_art_sents)
                    ext_list.append(ext_idx)
                    cur_idx += 1
                    print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                          cur_idx, n_data, cur_idx/n_data*100, timedelta(seconds=int(time()-start))
                    ), end='')
        print()

        # write files
        for file_idx, ext_ids in enumerate(ext_list):
            dec = []
            data_path = join(DATA_DIR, '{}/{}.json'.format(split, file_idx))
            with open(data_path) as f:
                data = json.loads(f.read())
            n_ext = 2 if data['source'] == 'CNN' else 3
            n_ext = min(n_ext, len(data['article']))
            for j in range(n_ext):
                sent_idx = ext_ids[j]
                dec.append(data['article'][sent_idx])
            with open(join(save_path, '{}.dec'.format(file_idx)), 'w') as f:
                for sent in dec:
                    print(sent, file=f)
        
        # evaluate current model
        print('Starting evaluating ROUGE !')
        dec_path = save_path
        ref_path = join(DATA_DIR, 'refs/{}'.format(split))
        ROUGE = eval_rouge(dec_path, ref_path)
        print(ROUGE)
        with open(join(args.path, 'ROUGE/{}.txt'.format(ckpts[i])), 'w') as f:
            print(ROUGE, file=f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the different encoder/decoder/emb_type'
    )
    parser.add_argument('--path', default='./result', help='root of the model')
    parser.add_argument('--mode', type=str, required=True, help='train/test')

    # model options
    parser.add_argument('--encoder', type=str, required=True,
                        help='BiLSTM/DeepLSTM/Transformer')
    parser.add_argument('--decoder', type=str, required=True, 
                        help='SL(Sequence Labeling)/PN(Pointer Network)')
    parser.add_argument('--emb_type', type=str, required=True,
                        help='W2V(word2vec)/BERT')
    parser.add_argument('--vsize', type=int, action='store', default=30000,
                        help='vocabulary size')
    parser.add_argument('--emb_dim', type=int, action='store', default=128,
                        help='the dimension of word embedding')
    parser.add_argument('--vocab_path', action='store',
                        help='use pretrained word2vec embedding')
    parser.add_argument('--conv_hidden', type=int, action='store', default=100,
                        help='the number of hidden units of Conv')
    parser.add_argument('--encoder_hidden', type=int, action='store', default=512,
                        help='the number of hidden units of encoder')
    parser.add_argument('--encoder_layer', type=int, action='store', default=1,
                        help='the number of layers of encoder')

    # length limit
    parser.add_argument('--max_word', type=int, action='store', default=100,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_sent', type=int, action='store', default=60,
                        help='maximun sentences in an article article')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=2e-5,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=0.5,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=3000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args = set_parameters(args)
    args.path = './{}_{}_{}'.format(args.encoder, args.decoder, args.emb_type)
    assert args.mode in ['train', 'test']

    if args.mode == 'train':
        train(args)
    else:
        test(args, 'test')
