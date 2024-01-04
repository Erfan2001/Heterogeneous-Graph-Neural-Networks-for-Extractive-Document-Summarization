import argparse

_DEBUG_FLAG_ = False


def pars_args():
    parser = argparse.ArgumentParser(description='ConGNN-SUM Model')
    root = "J:\\HSG"
    # Where to find data
    parser.add_argument('--data_dir', type=str,
                        default=f'{root}\datasets\cnndm',
                        help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str,
                        default=f'{root}\\ConGNN-SUM\\cache\\CNNDM',
                        help='The processed dataset directory')

    parser.add_argument('--embedding_path', type=str,
                        default=f'{root}\embeddings\glove.42B.300d.txt',
                        help='Path expression to external word embedding.')

    # Important settings
    parser.add_argument('--model', type=str, default='HSG', help='model structure[HSG|HDSG]')
    parser.add_argument('--test_model', type=str, default=r'J:\HSG\ConGNN-SUM\save\eval\bestmodel_0',
                        help='choose different model to test [multi/evalbestmodel/trainbestmodel/earlystop]')

    parser.add_argument('--use_pyrouge', action='store_true', default=False, help='use_pyrouge')



    parser.add_argument('--restore_model', type=str, default='None',
                        help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')

    # Hyperparameters
    parser.add_argument('--seed', type=int, default=666, help='set the random seed [default: 666]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. [default: 0]')
    parser.add_argument('--cuda', action='store_true', default=True, help='GPU or CPU [default: False]')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Size of vocabulary. [default: 50000]')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs [default: 20]')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration hop [default: 1]')

    parser.add_argument('--word_embedding', action='store_true', default=False,
                        help='whether to use Word embedding [default: True]')
    parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
    parser.add_argument('--embed_train', action='store_true', default=False,
                        help='whether to train Word embedding [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state [default: 128]')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of lstm layers [default: 2]')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='whether to use bidirectional LSTM [default: True]')
    parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature [default: 128]')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size [default: 64]')
    parser.add_argument('--gcn_hidden_size', type=int, default=128, help='hidden size [default: 64]')


    parser.add_argument('--ffn_inner_hidden_size', type=int, default=512,
                        help='PositionwiseFeedForward inner hidden size [default: 512]')
    parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
    parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1,
                        help='recurrent dropout prob [default: 0.1]')
    parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='attention dropout prob [default: 0.1]')
    parser.add_argument('--ffn_dropout_prob', type=float, default=0.1,
                        help='PositionwiseFeedForward dropout prob [default: 0.1]')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True,
                        help='use orthnormal init for lstm [default: True]')
    parser.add_argument('--sent_max_len', type=int, default=100,
                        help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--doc_max_timesteps', type=int, default=50,
                        help='max length of documents (max timesteps of documents)')

    # Training
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=False, help='learning rate descent')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='for gradient clipping max gradient normalization')

    parser.add_argument('-m', type=int, default=3, help='decode summary length')
    parser.add_argument('--save_label', action='store_true', default=True, help='require multihead attention')

    parser.add_argument('--limited', action='store_true', default=False, help='limited hypo length')
    parser.add_argument('--blocking', action='store_true', default=False, help='ngram blocking')


    parser.add_argument('--max_instances', type=int, default=None,help='max length of instances')
    parser.add_argument('--from_instances_index', type=int, default=0,help='from_instances_index')

    parser.add_argument('--use_cache_graph', type=bool, default=True,help='use cache')
    parser.add_argument('--fill_graph_cache', type=bool, default=False,help='use cache')

    args = parser.parse_args()

    return args


