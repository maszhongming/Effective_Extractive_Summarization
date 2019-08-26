import torch
import math
import numpy as np
from math import sqrt
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .rnn import lstm_encoder
from .util import sequence_mean, len_mask
from .attention import prob_normalize
from .DeepLSTM import DeepLSTM
from .TransformerEncoder import TransformerEncoder
from transformer.Models import get_sinusoid_encoding_table

from pytorch_pretrained_bert.modeling import BertModel

INI = 1e-2
MAX_ARTICLE_LEN = 512

class ConvSentEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout, emb_type):
        super().__init__()
        self._emb_type = emb_type
        if emb_type == 'W2V':
            self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(3, 6)])
        self._dropout = dropout
        self._grad_handle = None

    def forward(self, input_):
        if self._emb_type == 'W2V':
            emb_input = self._embedding(input_)
        else:
            emb_input = input_
        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=self.training)
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer, dropout, bidirectional):
        super().__init__()
        self._init_h = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        self._init_c = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        self._lstm = nn.LSTM(input_dim, n_hidden, n_layer,
                             dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_, in_lens=None):
        """ [batch_size, max_num_sent, input_dim] Tensor"""
        size = (self._init_h.size(0), input_.size(0), self._init_h.size(1))
        init_states = (self._init_h.unsqueeze(1).expand(*size),
                       self._init_c.unsqueeze(1).expand(*size))
        lstm_out, _ = lstm_encoder(
            input_, self._lstm, in_lens, init_states)
        return lstm_out.transpose(0, 1)

    @property
    def input_size(self):
        return self._lstm.input_size

    @property
    def hidden_size(self):
        return self._lstm.hidden_size

    @property
    def num_layers(self):
        return self._lstm.num_layers

    @property
    def bidirectional(self):
        return self._lstm.bidirectional

class LSTMPointerNet(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop, bidirectional=True):
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = torch.Tensor(input_dim)
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        init.uniform_(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(
            input_dim, n_hidden, n_layer,
            bidirectional=False, dropout=dropout
        )
        self._lstm_cell = None

        # attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        init.uniform_(self._attn_v, -INI, INI)

        # hop parameters
        self._hop_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._hop_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._hop_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._hop_wm)
        init.xavier_normal_(self._hop_wq)
        init.uniform_(self._hop_v, -INI, INI)
        self._n_hop = n_hop

    def forward(self, attn_mem, mem_sizes, lstm_in):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""
        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem, mem_sizes)
        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        query, final_states = self._lstm(lstm_in, lstm_states)
        query = query.transpose(0, 1)
        for _ in range(self._n_hop):
            query = LSTMPointerNet.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
        output = LSTMPointerNet.attention_score(
            attn_feat, query, self._attn_v, self._attn_wq)
        return output  # unormalized extraction logit

    def extract(self, attn_mem, mem_sizes, k):
        """extract k sentences, decode only, batch_size==1"""
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem, mem_sizes)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.get_device())
        extracts = []
        for _ in range(k):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            score = LSTMPointerNet.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            score = score.squeeze()
            for e in extracts:
                score[e] = -1e6
            ext = score.max(dim=0)[1].item()
            extracts.append(ext)
            lstm_states = (h, c)
            lstm_in = attn_mem[:, ext, :]
        return extracts

    def _prepare(self, attn_mem, mem_sizes):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))
        bs = attn_mem.size(0)
        n_l, d = self._init_h.size()
        size = (n_l, bs, d)
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous())
        d = self._init_i.size(0)

        # random
        # init_i = self._init_i.unsqueeze(0).unsqueeze(1).expand(bs, 1, d)
        
        # last state
        # init_i = attn_mem[:, -1, :].unsqueeze(1).expand(bs, 1, d)
        
        # max pooling
        init_i = LSTMPointerNet.max_pooling(attn_mem, mem_sizes)

        # mean pooling
        # init_i = LSTMPointerNet.mean_pooling(attn_mem, mem_sizes)
        
        return attn_feat, hop_feat, lstm_states, init_i

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2)  # [B, Nq, Ns, D]
        
        score = torch.matmul(
            torch.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]

        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector"""
        score = LSTMPointerNet.attention_score(attention, query, v, w)

        if mem_sizes is None: # decode
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.get_device()).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)

        output = torch.matmul(norm_score, attention)
        return output
    
    @staticmethod
    def mean_pooling(attn_mem, mem_sizes):
        if mem_sizes is None: # decode
            lens = torch.Tensor([attn_mem.size(1)]).cuda()
        else:
            lens = torch.Tensor(mem_sizes).unsqueeze(1).cuda()
        init_i = torch.sum(attn_mem, dim=1) / lens
        init_i = init_i.unsqueeze(1)
        return init_i
    
    @staticmethod
    def max_pooling(attn_mem, mem_sizes):
        if mem_sizes is not None:
            # not in decode
            B, Ns = attn_mem.size(0), attn_mem.size(1)
            mask = torch.ByteTensor(B, Ns).cuda()
            mask.fill_(0)
            for i, l in enumerate(mem_sizes):
                mask[i, :l].fill_(1)
            mask = mask.unsqueeze(-1)
            attn_mem = attn_mem.masked_fill(mask == 0, -1e18)
        init_i = attn_mem.max(dim=1, keepdim=True)[0]
        return init_i
        

class Summarizer(nn.Module):
    """ Different encoder/decoder/embedding type """
    def __init__(self, encoder, decoder, emb_type, emb_dim, vocab_size, 
                 conv_hidden, encoder_hidden, encoder_layer, 
                 isTrain=True, n_hop=1, dropout=0.0):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._emb_type = emb_type

        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout, emb_type)
        
        # BERT
        if emb_type == 'BERT':
            self._bert = BertModel.from_pretrained(
                         '/path/to/uncased_L-24_H-1024_A-16')
            self._bert.eval()
            for p in self._bert.parameters():
                p.requires_grad = False
            self._bert_w = nn.Linear(1024*4, emb_dim)

        # Sentence Encoder
        if encoder == 'BiLSTM':
            enc_out_dim = encoder_hidden * 2 # bidirectional
            self._art_enc = LSTMEncoder(
                3*conv_hidden, encoder_hidden, encoder_layer,
                dropout=dropout, bidirectional=True
            )
        elif encoder == 'Transformer':
            enc_out_dim = encoder_hidden
            self._art_enc = TransformerEncoder(
                3*conv_hidden, encoder_hidden, encoder_layer, decoder)
            
            self._emb_w = nn.Linear(3*conv_hidden, encoder_hidden)
            self.sent_pos_embed = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(1000, enc_out_dim, padding_idx=0), freeze=True)
        elif encoder == 'DeepLSTM':
            enc_out_dim = encoder_hidden
            self._isTrain = isTrain
            self._art_enc = DeepLSTM(
                3*conv_hidden, encoder_hidden, encoder_layer, 0.1)

        # Decoder
        decoder_hidden = encoder_hidden
        decoder_layer = encoder_layer
        if decoder == 'PN':
            self._extractor = LSTMPointerNet(
                enc_out_dim, decoder_hidden, decoder_layer,
                dropout, n_hop
            )
        else:
            self._ws = nn.Linear(enc_out_dim, 2)
            

    def forward(self, article_sents, sent_nums, target):
        enc_out = self._encode(article_sents, sent_nums)

        if self._decoder == 'PN':
            bs, nt = target.size()
            d = enc_out.size(2)
            ptr_in = torch.gather(
                enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d)
            )
            output = self._extractor(enc_out, sent_nums, ptr_in)
        
        else:
            bs, seq_len, d = enc_out.size()
            output = self._ws(enc_out)
            assert output.size() == (bs, seq_len, 2)

        return output

    def extract(self, article_sents, sent_nums=None, k=4):
        enc_out = self._encode(article_sents, sent_nums)
        
        if self._decoder == 'PN':
            extract = self._extractor.extract(enc_out, sent_nums, k)
        else:
            seq_len = enc_out.size(1)
            output = self._ws(enc_out)
            assert output.size() == (1, seq_len, 2)
            _, indices = output[:, :, 1].sort(descending=True)
            extract = []
            for i in range(k):
                extract.append(indices[0][i].item())

        return extract

    def _encode(self, article_sents, sent_nums):
        
        hidden_size = self._art_enc.input_size

        if sent_nums is None:  # test-time excode only
            if self._emb_type == 'W2V':
                enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
            else:
                enc_sent = self._article_encode(article=article_sents[0], 
                           device=article_sents[0].device).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            if self._emb_type == 'W2V':
                enc_sents = [self._sent_enc(art_sent)
                            for art_sent in article_sents]
            else:
                enc_sents = [self._article_encode(article=article, device=article.device)
                            for article in article_sents]
            def zero(n, device):
                z = torch.zeros(n, hidden_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.get_device())], dim=0)
                   if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        
        # Input for different encoder
        if self._encoder == 'BiLSTM':
            output = self._art_enc(enc_sent, sent_nums)

        elif self._encoder == 'Transformer':
            batch_size, seq_len = enc_sent.size(0), enc_sent.size(1)

            # prepare mask
            if sent_nums != None:
                input_len = len_mask(sent_nums, enc_sent.get_device()).float() # [batch_size, seq_len]
            else:
                input_len = torch.ones(batch_size, seq_len).float().cuda()

            attn_mask = input_len.eq(0.0).unsqueeze(1).expand(batch_size, 
                        seq_len, seq_len).cuda() # [batch_size, seq_len, seq_len]
            non_pad_mask = input_len.unsqueeze(-1).cuda()  # [batch, seq_len, 1]

            # add postional embedding
            if sent_nums != None:
                sent_pos = torch.LongTensor([np.hstack((np.arange(1, doclen + 1), 
                           np.zeros(seq_len - doclen))) for doclen in sent_nums]).cuda()
            else:
                sent_pos = torch.LongTensor([np.arange(1, seq_len + 1)]).cuda()

            inputs = self._emb_w(enc_sent) + self.sent_pos_embed(sent_pos)

            assert attn_mask.size() == (batch_size, seq_len, seq_len)
            assert non_pad_mask.size() == (batch_size, seq_len, 1)
            
            output = self._art_enc(inputs, non_pad_mask, attn_mask)
        
        elif self._encoder == 'DeepLSTM':
            batch_size, seq_len = enc_sent.size(0), enc_sent.size(1)
            inputs = [enc_sent.transpose(0, 1)]
            
            # prepare mask
            if sent_nums != None:
                inputs_mask = [len_mask(sent_nums, enc_sent.get_device()).transpose(0, 1).unsqueeze(-1)]
            else:
                inputs_mask = [torch.ones(seq_len, batch_size, 1).cuda()]
            
            for _ in range(self._art_enc.num_layers):
                inputs.append([None])
                inputs_mask.append([None])
            
            assert inputs[0].size() == (seq_len, batch_size, hidden_size)
            assert inputs_mask[0].size() == (seq_len, batch_size, 1)

            output = self._art_enc(inputs, inputs_mask, self._isTrain)

        return output

    def _article_encode(self, article, device, pad_idx=0):
        sent_num, sent_len = article.size()
        tokens_id = [101] # [CLS]
        for i in range(sent_num):
            for j in range(sent_len):
                if article[i][j] != pad_idx:
                    tokens_id.append(article[i][j])
                else:
                    break
        tokens_id.append(102) # [SEP]
        input_mask = [1] * len(tokens_id)
        total_len = len(tokens_id) - 2
        while len(tokens_id) < MAX_ARTICLE_LEN:
            tokens_id.append(0)
            input_mask.append(0)

        assert len(tokens_id)  == MAX_ARTICLE_LEN
        assert len(input_mask) == MAX_ARTICLE_LEN

        input_ids = torch.LongTensor(tokens_id).unsqueeze(0).to(device)
        input_mask = torch.LongTensor(input_mask).unsqueeze(0).to(device)
        
        # concat last 4 layers
        out, _ = self._bert(input_ids, token_type_ids=None, attention_mask=input_mask)
        out = torch.cat([out[-1], out[-2], out[-3], out[-4]], dim=-1)

        assert out.size() == (1, MAX_ARTICLE_LEN, 4096)
        
        emb_out = self._bert_w(out).squeeze(0)
        emb_dim = emb_out.size(-1)

        emb_input = torch.zeros(sent_num, sent_len, emb_dim).to(device)
        cur_idx = 1 # after [CLS]
        for i in range(sent_num):
            for j in range(sent_len):
                if article[i][j] != pad_idx:
                    emb_input[i][j] = emb_out[cur_idx]
                    cur_idx += 1
                else:
                    break
        assert cur_idx - 1 == total_len

        cnn_out = self._sent_enc(emb_input)
        assert cnn_out.size() == (sent_num, 300) # 300 = 3 * conv_hidden

        return cnn_out

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)

