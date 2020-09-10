# Effective_Extractive_Summarization
Code for ACL 2019 paper (oral):
*[Searching for Effective Neural Extractive Summarization: What Works and What's Next](https://arxiv.org/abs/1907.03491)*

If you use our code or data, please cite our paper:
```
@inproceedings{zhong2019searching,
  title={Searching for Effective Neural Extractive Summarization: What Works and What’s Next},
  author={Zhong, Ming and Liu, Pengfei and Wang, Danqing and Qiu, Xipeng and Huang, Xuan-Jing},
  booktitle={Proceedings of the 57th Conference of the Association for Computational Linguistics},
  pages={1049--1058},
  year={2019}
}

```

## Dependencies
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.1.0
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [cytoolz](https://github.com/pytoolz/cytoolz)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [pyrouge](https://github.com/bheinzerling/pyrouge)
- [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-transformers) 0.6.1
	- now is pytorch-transformers, you can use *pip* to install pytorch-pretrained-bert (0.6.1)
	- You should download the BERT model(bert-large-uncased) and convert it to a pytorch version, get a folder called `uncased_L-24_H-1024_A-16`
	
All code only supports running on *Linux*.

## Data

We have already processed CNN/DailyMail dataset, you can download it through [this link](https://drive.google.com/open?id=1QB9hVPF_YkJslaX4INnUZGS9OVL1Pr3O), unzip and store it in the current path (contrains train, val, test, refs, word2vec folders and vocab_cnt.pkl, you should put them in `./CNNDM`)

## Path

You should fill in the three paths in the files before running the code.
1. path to RELEASE-1.5.5 (evaluate.py line 14), example: `/home/ROUGE/RELEASE-1.5.5`
2. path to vocab.txt (decoding.py line 67 and data/batcher.py line 13), example: `/home/pretrain_model/uncased_L-24_H-1024_A-16/vocab.txt`
3. path to BERT model (model/extract.py line 255), example: `/home/pretrain_model/uncased_L-24_H-1024_A-16`

## Train

We currently provide a variety of options to combine into a model. For the encoder, we provide **BiLSTM/Transformer/DeepLSTM**. For the decoder, we provide **Sequence Labeling/Pointer Network**. For the type of word embedding, we provide **Word2Vec/BERT**.
We only tested the code on the GPU, and we strongly recommend using the GPU to train your model because of the long training time.

To run BiLSTM + Pointer Network + Word2Vec model, run

```
CUDA_VISIBLE_DEVICES=0 python main.py --mode=train --encoder=BiLSTM --decoder=PN --emb_type=W2V
```

To run Transformer + Sequence Labeling + Word2Vec model, run

```
CUDA_VISIBLE_DEVICES=0 python main.py --mode=train --encoder=Transformer --decoder=SL --emb_type=W2V
```

To run DeepLSTM + Pointer Network + BERT model (models with BERT have a long training time), run

```
CUDA_VISIBLE_DEVICES=0 python main.py --mode=train --encoder=DeepLSTM --decoder=PN --emb_type=BERT
```

You can try any other combination to train your own model.

## Test

After completing the training process, you can test the five best models and obtain ROUGE score by the following instructions.
You only need to switch mode to test, leaving other commands unchanged.

For example, when you test BiLSTM + Pointer Network + Word2Vec model, run

```
CUDA_VISIBLE_DEVICES=0 python main.py --mode=test --encoder=BiLSTM --decoder=PN --emb_type=W2V
```
The results will be printed on the screen and saved in the `BiLSTM_PN_W2V` folder.


## Output

You can find the outputs produced by our different models in this paper on CNN/DailyMail through [this link](https://drive.google.com/open?id=1VuZp2Wq3TH4kaXzuID34_e6VkBxa5Gkf).

## Note
1. Part of our code uses the the implementation of [fast_abs_rl](https://github.com/ChenRocks/fast_abs_rl) and [Transformer](https://github.com/jadore801120/attention-is-all-you-need-pytorch). Thanks for their work!
2. For the code of reinforcement learning, we use the implementation in [fast_abs_rl](https://github.com/ChenRocks/fast_abs_rl). The only difference is that we changed the parameter *reward* in their file *train_full_rl.py* to ROUGE-1-precision and the parameter *stop* to 8.5. If you are interested in the implementation of this part, please directly refer to their code.
