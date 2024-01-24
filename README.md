# ConGNN-SUM: Contextualized-Heterogeneous-Graph-Neural-Networks-for-Extractive-Document-Summarization

# Methodolgy

![image](https://github.com/Erfan2001/Heterogeneous-Graph-Neural-Networks-for-Extractive-Document-Summarization/assets/69463039/d7be478c-5573-434f-a5a8-0466ed863559)


# Description of Project:

The Mail Daily/CNN dataset comprises numerous documents, each housing multiple sentences. In line with the exemplar article titled "Heterogeneous Graph Neural Networks for Summarizing Extractive Documents," we limit our consideration to a maximum of 50 sentences within each document, disregarding the remaining sentences. Furthermore, we restrict our focus to the first 100 words within each sentence, disregarding the remainder. Our approach involves converting documents into sentences and sentences into words. Using word embeddings initialized with Glove weights, we obtain 300-dimensional vectors for each word. In a specific case, we generate a 100x300 vector for each sentence, apply convolutional neural networks (CNN) with 2, 3, 4, 5, 6, and 7-dimensional kernels to capture n-grams, and produce -2 grams, -3 grams, -4 grams, -5 grams, -6 grams, and -7 grams, thereby influencing the content representations. Subsequently, we derive distinct features for each n-gram and concatenate them. This process yields a 100x300 vector for each n-gram. These vectors undergo processing through a bidirectional Long Short-Term Memory (LSTM-Bi) neural network, which results in five hidden states that are effectively representations of word vectors. In summary, a 100-word sentence, with each word represented as a 300-dimensional vector, enters the LSTM-Bi, producing output vectors of dimension 64. Ultimately, following linear transformations for each document, the output takes the form of a 50x100x64 vector.

Moving forward, we construct a graph utilizing the obtained words and sentences. Each node within the graph corresponds to a word, and we designate one node for each sentence. For word nodes, we establish connections to all words within a sentence, with edge weights determined through the IDF-TF method. We assign the same embedding to each node. In essence, each graph is document-specific, featuring 300-dimensional words and 64-dimensional sentences. These graphs are subsequently input into a graph network, which propagates message-based features among neighboring nodes, updating their final states. Graph attention networks are employed to refine semantic node representations. As part of this process, characteristics of word and sentence nodes become intertwined, with message exchange occurring among nodes and their neighbors. While individual sentences may not directly relate to one another, they share common words, driving the similarity of sentences conveying similar concepts. The method for node communication involves injecting word node features into the sentence nodes and vice versa, iteratively influencing each other. This multi-stage approach allows for the adjustment of sentences based on word characteristics and fosters similarity between sentences in the absence of direct connections. The number of iterations can be controlled to modulate the influence of feature parameters.

Some codes are borrowed from [PG](https://github.com/abisee/pointer-generator) and [Transformer](https://github.com/jadore801120/attention-is-all-you-need-pytorch). Thanks for their work.

### Dependency 

- python 3.5+
- [PyTorch](https://pytorch.org/) 1.0+
- [DGL](http://dgl.ai) 0.4
- [rouge](https://github.com/pltrdy/rouge) 1.0.0
  - A full Python Implementation of the ROUGE Metric which is used in validation phase
- [pyrouge](https://github.com/bheinzerling/pyrouge) 0.1.3

- others
  - nltk
  - numpy
  - sklearn



## Data

We have preprocessed **CNN/DailyMail** dataset for TF-IDF features used in the graph creation, which you can find [here](https://drive.google.com/open?id=1oIYBwmrB9_alzvNDBtsMENKHthE9SW9z).

For **CNN/DailyMail**, we also provide the json-format datasets in [this link](https://drive.google.com/open?id=1JW033KefyyoYUKUFj6GqeBFZSHjksTfr).

The example looks like this:

```
{
  "text":["deborah fuller has been banned from keeping animals ... 30mph",...,"a dog breeder and exhibitor... her dogs confiscated"],
  "summary":["warning : ... at a speed of around 30mph",... ,"she was banned from ... and given a curfew "],
  "label":[1,3,6]
}
```

and each line in the file is an example.  For the *text* key, the value can be list of string (single-document) or list of list of string (multi-document). The example in training set can ignore the *summary* key since we only use *label* during the training phase. All strings need be lowercase and tokenized by [Stanford Tokenizer](https://nlp.stanford.edu/software/tokenizer.shtml), and  ***nltk.sent_tokenize*** is used to get sentences.

After getting the standard json format, you can prepare the dataset for the graph by ***PrepareDataset.sh*** in the project directory. The processed files will be put under the ***cache*** directory.

The default file names for training, validation and test are: *train.label.jsonl*, *val.label.jsonl* and *test.label.jsonl*. If you would like to use other names, please change the corresponding names in  ***PrepareDataset.sh***,  Line 321-322 in ***train.py*** and Line 188 in ***evaluation.py***. (Default names is recommended)



## Train

For training, you can run commands like this:

```shell
python train.py --cuda --gpu 0 --data_dir <data/dir/of/your/json-format/dataset> --cache_dir <cache/directory/of/graph/features> --embedding_path <glove_path> --model [HSG|HDSG] --save_root <model path> --log_root <log path> --lr_descent --grad_clip -m 3
```



We also provide our checkpoints on **CNN/DailyMail**, **NYT50** and **Multi-News** in [this link](https://drive.google.com/open?id=16wA_JZRm3PrDJgbBiezUDExYmHZobgsB). Besides, the outputs can be found [here](https://drive.google.com/open?id=1VArOyIbGO8ayW0uF8RcmN4Lh2DDtmcQz)(NYT50 has been removed due to its license).



## Test

For evaluation, the command may like this:

```shell
python evaluation.py --cuda --gpu 0 --data_dir <data/dir/of/your/json-format/dataset> --cache_dir <cache/directory/of/graph/features> --embedding_path <glove_path>  --model [HSG|HDSG] --save_root <model path> --log_root <log path> -m 3 --test_model multi --use_pyrouge
```

Some options:

- *use_pyrouge*: whether to use pyrouge for evaluation. Default is **False** (which means rouge).
  - Please change Line17-18 in ***tools/utils.py*** to your own ROUGE path and temp file path.
- *limit*: whether to limit the output to the length of gold summaries. This option is only set for evaluation on NYT50 (which uses ROUGE-recall instead of ROUGE-f). Default is **False**.
- *blocking*: whether to use Trigram blocking. Default is **False**.
- save_label: only save label and do not calculate ROUGE. Default is **False**.



To load our checkpoint for evaluation, you should put it under the ***save_root/eval/*** and make the name for test_model to start with ***eval***. For example, if your save_root is "*checkpoints*", then the checkpoint "*cnndm.ckpt*" should be put under "*checkpoints/eval*" and the test_model is *evalcnndm.ckpt*.



## ROUGE Installation

In order to get correct ROUGE scores, we recommend using the following commands to install the ROUGE environment:

```shell
sudo apt-get install libxml-perl libxml-dom-perl
pip install git+git://github.com/bheinzerling/pyrouge
export PYROUGE_HOME_DIR=the/path/to/RELEASE-1.5.5
pyrouge_set_rouge_path $PYROUGE_HOME_DIR
chmod +x $PYROUGE_HOME_DIR/ROUGE-1.5.5.pl
```

You can refer to https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5 for RELEASE-1.5.5 and remember to build Wordnet 2.0 instead of 1.6 in RELEASE-1.5.5/data:

```shell
cd $PYROUGE_HOME_DIR/data/WordNet-2.0-Exceptions/
./buildExeptionDB.pl . exc WordNet-2.0.exc.db
cd ../
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
```
