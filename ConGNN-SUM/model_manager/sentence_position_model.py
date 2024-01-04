import torch
import torch.nn as nn
import dgl
from model_manager.hsg_model import HSumGraph

position_probabilities = [0.199290, 0.228100, 0.222377, 0.222499, 0.192609, 0.153802, 0.123783, 0.101765, 0.088789,
                          0.078482, 0.071153, 0.066298, 0.060599, 0.055165, 0.049167, 0.043374, 0.035558, 0.028654,
                          0.021482, 0.016149, 0.012366, 0.008329, 0.005747, 0.004382, 0.003083, 0.002205, 0.001501,
                          0.001024, 0.000731, 0.000470, 0.000293, 0.000136, 0.000104, 0.000098, 0.000052, 0.000014,
                          0.000010, 0.000014, 0.000003] + [0.000003] * 20
position_probabilities = torch.Tensor([[1 - x, x] for x in position_probabilities])


class RnnHSGModel(nn.Module):
    def __init__(self, hps, embed):
        super(RnnHSGModel, self).__init__()
        self.hps = hps
        self.HSG = HSumGraph(embed=embed, hps=hps)
        self.rnn = torch.nn.LSTM(hps.hidden_size, 128, 1, bias=True)
        self.classifier = torch.nn.Sequential(torch.nn.Linear(128, 2), torch.nn.Sigmoid())
        self.position_probabilities = position_probabilities.to(hps.device)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.to(hps.device)

    def forward(self, graph):
        hsg_p, sent_features = self.HSG(graph)
        # probabilities, sent_features = self.HSG(graph)
        graph_list = dgl.unbatch(graph)
        indices = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graph_list]
        rnn_results = torch.Tensor().to(self.hps.device)
        batch_position_probabilities = torch.Tensor().to(self.hps.device)
        for sentence_vector in torch.split(sent_features, indices):
            new_rnn_result = self.classifier(self.rnn(sentence_vector)[0])
            # new_result = (self.softmax(hsg_p) +
            #               self.position_probabilities[:sentence_vector.shape[0], :] + self.softmax(rnn_result)) / 3
            rnn_results = torch.cat([rnn_results, new_rnn_result])
            batch_position_probabilities = torch.cat(
                [batch_position_probabilities, self.position_probabilities[:sentence_vector.shape[0], :]])

        return self.softmax(hsg_p) + batch_position_probabilities + self.softmax(rnn_results) / 3
        # return self.sentence_level_model(sent_features, probabilities)
