import torch
import torch.nn as nn
import dgl
from model_manager.hsg_model import HSumGraph
from torch.nn.functional import normalize


class TextRepresentationHSGModel(HSumGraph):
    def __init__(self, hps, embed):
        super(TextRepresentationHSGModel, self).__init__(embed=embed, hps=hps)
        self.rnn = torch.nn.LSTM(hps.hidden_size, hps.hidden_size, 1, bias=True, batch_first=True)
        self.classifier = torch.nn.Sequential(nn.Linear(2 * self.n_feature, 64), nn.ELU(), nn.Linear(64, 2))
        self.to(hps.device)

    def forward(self, graph):

        word_feature = self.set_wnfeature(graph)  # [wnode, embed_size]

        sent_feature = self.n_feature_proj(self.set_snfeature(graph))  # [snode, n_feature_size]

        # the start state
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        for i in range(self._n_iter):
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)

        i = 0
        text_feature = torch.Tensor().to(self._hps.device)
        for g in dgl.unbatch(graph):
            snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            sentence_len = len(snode_id)
            text_sent_feature = sent_feature[i:i + sentence_len, :]
            text_feature = torch.cat([text_feature, self.rnn(text_sent_feature.unsqueeze(dim=0))[1][1].squeeze().repeat(
                sentence_len).reshape((sentence_len, -1))])
            i += sentence_len
        result = self.classifier(torch.cat([sent_state, text_feature], 1))
        # result = self.wh(sent_state)

        return result


class RnnTextRepresentationModel(nn.Module):
    def __init__(self, hps, embed):
        super(RnnTextRepresentationModel, self).__init__()
        self.hps = hps
        self.HSG = HSumGraph(embed=embed, hps=hps)
        self.rnn_text_representation = torch.nn.LSTM(hps.hidden_size, hps.hidden_size, 2, bias=True, batch_first=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hps.hidden_size * 3, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 2),
            torch.nn.ELU()
        )
        self.dropout = torch.nn.Dropout(p=0.2)
        self.to(hps.device)

    def forward(self, graph):
        sent_features = self.HSG(graph)
        # probabilities, sent_features = self.HSG(graph)
        graph_list = dgl.unbatch(graph)
        indices = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graph_list]
        text_feature = torch.Tensor().to(self.hps.device)
        for sentence_vector in torch.split(sent_features, indices):
            sentence_len = sentence_vector.shape[0]
            new_text_feature = self.rnn_text_representation(sentence_vector.unsqueeze(dim=0))[1][1].reshape(-1)
            new_text_feature = new_text_feature.repeat(sentence_len).reshape((sentence_len, -1))
            text_feature = torch.cat([text_feature, new_text_feature])
        result = self.dropout(self.classifier(torch.cat([sent_features      , text_feature], 1)))

        return result
        # return self.sentence_level_model(sent_features, probabilities)
