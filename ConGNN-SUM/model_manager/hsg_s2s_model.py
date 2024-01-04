import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GATConv
from model_manager.hsg_model import HSumGraph

class HSumGraphWithS2SModel(nn.Module):

    def __init__(self, hps, embed):
        super(HSumGraphWithS2SModel, self).__init__()
        self.hps = hps
        self.HSG = HSumGraph(embed=embed, hps=hps)
        self.num_heads = 4
        self.s2s_gat_conv = GATConv(in_feats=hps.hidden_size, out_feats=hps.hidden_size, num_heads=self.num_heads)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hps.hidden_size * self.num_heads, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 2)
        )
        self.to(hps.device)

    def make_sentence_graph(self, graph):
        u, v = torch.Tensor([]), torch.Tensor([])
        last_index = 0

        graphs = dgl.unbatch(graph)
        for g in graphs:
            sentences = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            new_u = torch.Tensor(list(range(len(sentences) - 1))) + last_index
            new_v = torch.Tensor(list(range(1, len(sentences)))) + last_index
            u = torch.cat([u, new_u, new_v])
            v = torch.cat([v, new_v, new_u])
            last_index += len(sentences)

        return dgl.graph((list(u), list(v)))

    def forward(self, graph):
        sent_features = self.HSG(graph)
        sentence_graph = self.make_sentence_graph(graph).to(self.hps.device)
        sent_features = self.s2s_gat_conv(sentence_graph, sent_features)
        sent_features = sent_features.reshape(-1, self.num_heads * self.hps.hidden_size)
        result = self.classifier(sent_features)
        return result
