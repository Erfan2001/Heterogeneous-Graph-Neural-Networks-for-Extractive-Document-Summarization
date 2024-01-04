import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

from dgl.data.utils import save_graphs
from nltk.corpus import stopwords
import time
import json
from collections import Counter
import numpy as np
import torch
import torch.utils.data
from tools.logger import *
import dgl
from dgl.data.utils import load_graphs

FILTER_WORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTER_WORD.extend(punctuations)


class Example(object):
    """Class representing a train/val/test example for single-document extractive summarization."""

    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        :param article_sents: list(strings) for single document or list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param vocab: Vocabulary object
        :param sent_max_len: int, max length of each sentence
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """

        self.sent_max_len = sent_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []

        # Store the original strings
        self.original_article_sents = article_sents
        self.original_abstract = "\n".join(abstract_sents)

        # Process the article
        if isinstance(article_sents, list) and isinstance(article_sents[0], list):  # multi document
            self.original_article_sents = []
            for doc in article_sents:
                self.original_article_sents.extend(doc)
        for sent in self.original_article_sents:
            article_words = sent.split()
            self.enc_sent_len.append(len(article_words))  # store the length before padding
            self.enc_sent_input.append([vocab.word2id(w.lower()) for w in
                                        article_words])  # list of word ids; OOVs are represented by the id for UNK token
        self._pad_encoder_input(vocab.word2id('[PAD]'))

        # Store the label
        self.label = label
        label_shape = (len(self.original_article_sents), len(label))  # [N, len(label)]
        # label_shape = (len(self.original_article_sents), len(self.original_article_sents))
        self.label_matrix = np.zeros(label_shape, dtype=int)
        if label != []:
            self.label_matrix[np.array(label), np.arange(
                len(label))] = 1  # label_matrix[i][j]=1 indicate the i-th sent will be selected in j-th step

    def _pad_encoder_input(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return: 
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)


class Example2(Example):
    """Class representing a train/val/test example for multi-document extractive summarization."""

    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        :param article_sents: list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param vocab: Vocabulary object
        :param sent_max_len: int, max length of each sentence
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """

        super().__init__(article_sents, abstract_sents, vocab, sent_max_len, label)
        cur = 0
        self.original_articles = []
        self.article_len = []
        self.enc_doc_input = []
        for doc in article_sents:
            if len(doc) == 0:
                continue
            docLen = len(doc)
            self.original_articles.append(" ".join(doc))
            self.article_len.append(docLen)
            self.enc_doc_input.append(catDoc(self.enc_sent_input[cur:cur + docLen]))
            cur += docLen


class SummarizationDataSet(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self, data_path, vocab, filter_word_path, w2s_path, hps, graphs_dir=None):
        """ Initializes the ExampleSet with the path of data
        
        :param data_path: string; the path of data
        :param vocab: object;
        :param doc_max_time_steps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        """
        self.hps = hps
        self.vocab = vocab
        self.sent_max_len = hps.sent_max_len
        self.doc_max_timesteps = hps.doc_max_timesteps
        self.max_instance = hps.max_instances

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.example_list = read_json(data_path, max_instance=self.max_instance,
                                      from_instances_index=hps.from_instances_index)
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))
        self.size = len(self.example_list)

        logger.info("[INFO] reading filter word File %s", filter_word_path)
        tfidf_w = read_text(filter_word_path)
        self.filter_words = FILTER_WORD
        self.filter_ids = [vocab.word2id(w.lower()) for w in FILTER_WORD]
        self.filter_ids.append(vocab.word2id("[PAD]"))  # keep "[UNK]" but remove "[PAD]"
        logger.info("[INFO] Loading filter word File")
        filter_words = list(set(tfidf_w).intersection(set(vocab._word_to_id.keys())))[:5000]
        self.filter_words += filter_words
        self.filter_ids += [self.vocab._word_to_id[word] for word in filter_words]

        logger.info("[INFO] Loading word2sent TFIDF file from %s!" % w2s_path)
        self.w2s_tfidf = read_json(w2s_path, max_instance=self.max_instance,
                                   from_instances_index=hps.from_instances_index)
        self.graphs_dir = graphs_dir
        self.use_cache = self.hps.use_cache_graph
        self.fill_cache = self.hps.fill_graph_cache
        if self.fill_cache:
            self.cache_graphs()

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example

    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        N, m = label_m.shape
        if m < self.doc_max_timesteps:
            pad_m = np.zeros((N, self.doc_max_timesteps - m))
            return np.hstack([label_m, pad_m])
        return label_m

    def add_word_node(self, graph, input_id):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in input_id:
            for wid in sentid:
                if wid not in self.filter_ids and wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1

        w_nodes = len(nid2wid)

        graph.add_nodes(w_nodes)
        graph.set_n_initializer(dgl.init.zero_initializer)
        graph.ndata["unit"] = torch.zeros(w_nodes)
        graph.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
        graph.ndata["dtype"] = torch.zeros(w_nodes)

        return wid2nid, nid2wid

    def create_graph(self, input_pad, label, w2s_w):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, dtype=0
        """
        # noinspection DuplicatedCode
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.add_word_node(G, input_pad)
        w_nodes = len(nid2wid)

        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        G.set_e_initializer(dgl.init.zero_initializer)
        for i in range(N):
            c = Counter(input_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid in c.keys():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})

            # The two lines can be commented out if you use the code for your own training, since HSG does not use sent2sent edges.
            # However, if you want to use the released checkpoint directly, please leave them here.
            # Otherwise it may cause some parameter corresponding errors due to the version differences.
            G.add_edges(sent_nid, sentid2nid, data={"dtype": torch.ones(N)})
            G.add_edges(sentid2nid, sent_nid, data={"dtype": torch.ones(N)})
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

        return G

    def get_edges(self, i, input_pad_i, sentid2nid, w2s_w_i, wid2nid, N):
        edges = []
        c = Counter(input_pad_i)
        sent_nid = sentid2nid[i]
        sent_tfw = w2s_w_i
        for wid in c.keys():
            if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                tfidf = sent_tfw[self.vocab.id2word(wid)]
                tfidf_box = np.round(tfidf * 9)  # box = 10
                edges.append(
                    (wid2nid[wid], sent_nid, {"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])}))
                edges.append(
                    (sent_nid, wid2nid[wid], {"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])}))

        edges.append((sent_nid, sentid2nid, {"dtype": torch.ones(N)}))
        edges.append((sentid2nid, sent_nid, {"dtype": torch.ones(N)}))
        return edges

    def get_graph(self, index):
        item = self.get_example(index)
        input_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        label = self.pad_label_m(item.label_matrix)
        w2s_w = self.w2s_tfidf[index]
        G = self.create_graph(input_pad, label, w2s_w)
        return G

    def __getitem__(self, index):

        G = self.get_graph(index)

        return G, index

    def __getitems__(self, possibly_batched_index):
        data = []

        with ProcessPoolExecutor(max_workers=1) as executor:
            for index, item in enumerate(executor.map(self.__getitem__, possibly_batched_index)):
                data.append(item)

        return data

    def _make_graphs_and_save(self, index):
        print(f"make graph for index = {index}")
        t1 = time.time()
        graphs = [item[0] for item in
                  self.__getitems__(list(range(index, min(index + 256, len(self.example_list)))))]
        path = os.path.join(self.graphs_dir, f"{index + self.hps.from_instances_index}.bin")
        save_graphs(path, graphs)
        del graphs

        # save_graphs(filename=path, g_list=graphs)
        print(f"save graphs {index} to {index + 255} =>  {time.time() - t1}")

    def cache_graphs(self):
        processes = []
        for index in range(0, len(self.example_list), 256):
            p = multiprocessing.Process(target=self._make_graphs_and_save, args=(index,))
            processes.append(p)
            p.start()
        for process in processes:
            process.join()
            process.close()
            del process
        del processes
        logger.info("finish save graphs")

    def __len__(self):
        return self.size



class MultiSummarizationDataSet(SummarizationDataSet):
    """ Constructor: Dataset of example(object) for multiple document summarization"""

    def __init__(self, data_path, vocab, doc_max_time_steps, sent_max_len, filter_word_path, w2s_path, w2d_path):
        """ Initializes the ExampleSet with the path of data

        :param data_path: string; the path of data
        :param vocab: object;
        :param doc_max_time_steps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py)
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        :param w2d_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2dTFIDF.py)
        """

        super().__init__(data_path, vocab, doc_max_time_steps, sent_max_len, filter_word_path, w2s_path)

        logger.info("[INFO] Loading word2doc TFIDF file from %s!" % w2d_path)
        self.w2d_tfidf = read_json(w2d_path)

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example2(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        del e
        return example

    def MapSent2Doc(self, article_len, sentNum):
        sent2doc = {}
        doc2sent = {}
        sentNo = 0
        for i in range(len(article_len)):
            doc2sent[i] = []
            for j in range(article_len[i]):
                sent2doc[sentNo] = i
                doc2sent[i].append(sentNo)
                sentNo += 1
                if sentNo >= sentNum:
                    return sent2doc
        return sent2doc

    def create_graph(self, docLen, sent_pad, doc_pad, label, w2s_w, w2d_w):
        """ Create a graph for each document

        :param docLen: list; the length of each document in this example
        :param sent_pad: list(list), [sentnum, wordnum]
        :param doc_pad: list, [document, wordnum]
        :param label: list(list), [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}, for each sentence and each word, the tfidf between them
        :param w2d_w: dict(dict) {str: {str: float}}, for each document and each word, the tfidf between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
                document: unit=1, dtype=2
            edge:
                word2sent, sent2word: tffrac=int, dtype=0
                word2doc, doc2word: tffrac=int, dtype=0
                sent2doc: dtype=2
        """
        # add word nodes
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.add_word_node(G, sent_pad)
        w_nodes = len(nid2wid)

        # add sent nodes
        N = len(sent_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        ws_nodes = w_nodes + N

        # add doc nodes
        sent2doc = self.MapSent2Doc(docLen, N)
        article_num = len(set(sent2doc.values()))
        G.add_nodes(article_num)
        G.ndata["unit"][ws_nodes:] = torch.ones(article_num)
        G.ndata["dtype"][ws_nodes:] = torch.ones(article_num) * 2
        docid2nid = [i + ws_nodes for i in range(article_num)]

        # add sent edges
        for i in range(N):
            c = Counter(sent_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid, cnt in c.items():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    # w2s s2w
                    G.add_edge(wid2nid[wid], sent_nid,
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edge(sent_nid, wid2nid[wid],
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            # s2d
            docid = sent2doc[i]
            docnid = docid2nid[docid]
            G.add_edge(sent_nid, docnid, data={"tffrac": torch.LongTensor([0]), "dtype": torch.Tensor([2])})

        # add doc edges
        for i in range(article_num):
            c = Counter(doc_pad[i])
            doc_nid = docid2nid[i]
            doc_tfw = w2d_w[str(i)]
            for wid, cnt in c.items():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in doc_tfw.keys():
                    # w2d d2w
                    tfidf = doc_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edge(wid2nid[wid], doc_nid,
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edge(doc_nid, wid2nid[wid],
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})

        G.nodes[sentid2nid].data["words"] = torch.LongTensor(sent_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

        return G

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)
        sent_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        enc_doc_input = item.enc_doc_input
        article_len = item.article_len
        label = self.pad_label_m(item.label_matrix)

        G = self.create_graph(article_len, sent_pad, enc_doc_input, label, self.w2s_tfidf[index], self.w2d_tfidf[index])
        del item
        return G, index


class LoadHiExampleSet(torch.utils.data.Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.gfiles = [f for f in os.listdir(self.data_root) if f.endswith("graph.bin")]
        logger.info("[INFO] Start loading %s", self.data_root)

    def __getitem__(self, index):
        graph_file = os.path.join(self.data_root, "%d.graph.bin" % index)
        g, label_dict = load_graphs(graph_file)
        return g[0], index

    def __len__(self):
        return len(self.gfiles)


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res


def read_json(fname, max_instance=None, from_instances_index=None):
    data_list = []
    count = 0
    if from_instances_index is None:
        from_instances_index = 0
    print(from_instances_index)
    with open(fname, encoding="utf-8") as f:
        lines = f.readlines()
        for data in lines[from_instances_index:]:
            if max_instance is not None and count >= max_instance:
                return data_list
            data_list.append(json.loads(data))
            count += 1

    return data_list


def read_text(file_name):
    data = []
    with open(file_name, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data


def graph_collate_fn(samples):
    graphs, index = map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    return batched_graph, [index[idx] for idx in sorted_index]
