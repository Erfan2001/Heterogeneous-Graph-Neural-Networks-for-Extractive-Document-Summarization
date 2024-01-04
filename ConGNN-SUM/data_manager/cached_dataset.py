import os
import torch.utils.data
from dgl.data.utils import load_graphs

from data_manager.dataloader import Example, read_json


class CachedSummarizationDataSet(torch.utils.data.Dataset):

    def __init__(self, hps, data_path=None, vocab=None, graphs_dir=None, from_index=0, to_index=0):
        self.hps = hps
        self.sent_max_len = hps.sent_max_len
        self.doc_max_timesteps = hps.doc_max_timesteps
        self.max_instance = hps.max_instances
        self.graphs_dir = graphs_dir
        self.use_cache = self.hps.fill_graph_cache
        self.from_index = from_index
        self.to_index = to_index
        self.graph_index_from = 0
        self.graph_index_offset = 256
        root, _, files = list(os.walk(self.graphs_dir))[0]
        indexes = [int(item[:-4]) for item in files]
        max_index = max(indexes)
        # size = max(indexes) + self.graph_index_offset - 1
        g, label_dict = load_graphs(os.path.join(root, f"{max_index}.bin"))
        size = max_index + len(g) - 1
        if to_index is None:
            to_index = size
        max_instances = hps.max_instances if hps.max_instances else 288000
        self.size = min([to_index - from_index, max_instances, size])
        self.graphs = dict()
        self.load_HSG_graphs()
        self.example_list = None
        self.vocab = vocab
        self.data_path = data_path

    def fill_example_list(self):
        self.example_list = read_json(self.data_path, max_instance=self.max_instance,
                                      from_instances_index=self.hps.from_instances_index)

    def get_example(self, index):
        if self.example_list is None:
            self.fill_example_list()

        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example

    def load_HSG_graphs(self):
        graphs, _ = load_graphs(os.path.join(self.graphs_dir, f"{self.graph_index_from}.bin"))
        for i, graph in enumerate(graphs):
            self.graphs[self.graph_index_from + i] = graph

    def get_graph(self, index):
        if index not in self.graphs.keys():
            self.graph_index_from = (index // self.graph_index_offset) * self.graph_index_offset
            self.load_HSG_graphs()

        return self.graphs[index]

    def __getitem__(self, index):
        try:
            G = self.get_graph(index)

            return G, index
        except Exception as e:
            print(f"EXCEPTION => {e}")
            return None

    def __getitems__(self, possibly_batched_index):
        result = []
        for index in possibly_batched_index:
            item = self.__getitem__(self.from_index + index)
            if item is not None:
                result.append(item)

        return result

    def __len__(self):
        return self.size
