import torch
import torch.nn as nn
import dgl
from model_manager.hsg_model import HSumGraph


class RnnHSGModel(nn.Module):
    def __init__(self, hps, embed):
        super(RnnHSGModel, self).__init__()
        self.hps = hps
        self.HSG = HSumGraph(embed=embed, hps=hps)
        # self.rnn = torch.nn.LSTM(hps.hidden_size, hps.hidden_size, 4, bias=True,batch_first=True,bidirectional=True)
        self.rnn = torch.nn.GRU(hps.hidden_size, hps.hidden_size, 4, bias=True,batch_first=True,bidirectional=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2*hps.hidden_size,hps.hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hps.hidden_size,2),
            torch.nn.ELU()
        )
        self.to(hps.device)

    def forward(self, graph):
        sent_features = self.HSG(graph)
        # probabilities, sent_features = self.HSG(graph)
        graph_list = dgl.unbatch(graph)
        indices = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graph_list]
        rnn_results = torch.Tensor().to(self.hps.device)
        for sentence_vector in torch.split(sent_features, indices):
            rnn_result = self.rnn(sentence_vector.unsqueeze(dim=0))[0].squeeze()
            rnn_results = torch.cat([rnn_results,rnn_result])
        result = self.classifier(rnn_results)

        return result
        # return self.sentence_level_model(sent_features, probabilities)

