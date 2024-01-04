from torch.utils.data import BatchSampler

from data_manager.dataloader import SummarizationDataSet, graph_collate_fn
from data_manager.cached_dataset import CachedSummarizationDataSet
import torch


def make_dataloader(data_file, vocab, hps, filter_word, w2s_path, graphs_dir=None, from_index=0, to_index=None,
                    shuffle=False):
    if hps.use_cache_graph:
        dataset = CachedSummarizationDataSet(hps=hps, graphs_dir=graphs_dir, vocab=vocab, data_path=data_file,
                                             from_index=from_index, to_index=to_index)
    else:
        dataset = SummarizationDataSet(data_path=data_file, vocab=vocab, filter_word_path=filter_word,
                                       w2s_path=w2s_path, hps=hps, graphs_dir=graphs_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=shuffle, num_workers=0,
                                         collate_fn=graph_collate_fn)
    del dataset
    return loader
