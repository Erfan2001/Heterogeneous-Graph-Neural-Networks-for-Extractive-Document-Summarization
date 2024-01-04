import argparse
import datetime
import os
import time

import torch
from rouge import Rouge

from model_manager.model import Model
from Tester import SLTester
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools import utils
from tools.logger import *
from config import pars_args
from utils import set_device
from data_manager import data_loaders


def load_test_model(model, model_name, eval_dir, save_root):
    """ choose which model will be loaded for evaluation """
    path = os.path.join(save_root, model_name)
    model.load_state_dict(torch.load(path))
    return model

    if model_name.startswith('eval'):
        bestmodel_load_path = os.path.join(eval_dir, model_name[4:])
    elif model_name.startswith('train'):
        train_dir = os.path.join(save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, model_name[5:])
    elif model_name == "earlystop":
        train_dir = os.path.join(save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, 'earlystop')
    else:
        logger.error("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
        raise ValueError("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
    if not os.path.exists(bestmodel_load_path):
        print(bestmodel_load_path)
        logger.error("[ERROR] Restoring %s for testing...The path %s does not exist!", model_name, bestmodel_load_path)
        return None
    logger.info("[INFO] Restoring %s for testing...The path is %s", model_name, bestmodel_load_path)

    model.load_state_dict(torch.load(bestmodel_load_path))

    return model


def run_test(model, dataset, loader, model_name, hps):
    test_dir = os.path.join(hps.save_root, "test")  # make a subdir of the root dir for eval data
    eval_dir = os.path.join(hps.save_root, "eval")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(eval_dir):
        logger.exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it.", eval_dir)
        raise Exception(f"[Error] eval_dir {eval_dir} doesn't exist. Run in train mode to create it.")

    model = load_test_model(model, model_name, eval_dir, hps.save_root)
    model.eval()

    iter_start_time = time.time()
    with torch.no_grad():
        logger.info("[Model] Sequence Labeling!")
        tester = SLTester(model, hps.m, limited=hps.limited, test_dir=test_dir)

        for i, (G, index) in enumerate(loader):
            G = G.to(hps.device)
            tester.evaluation(G, index, dataset, blocking=hps.blocking)

    running_avg_loss = tester.running_avg_loss

    logger.info("The number of pairs is %d", tester.rougePairNum)
    if not tester.rougePairNum:
        logger.error("During testing, no hyps is selected!")
        sys.exit(1)

    if hps.use_pyrouge:
        if isinstance(tester.refer[0], list):
            logger.info("Multi Reference summaries!")
            scores_all = utils.pyrouge_score_all_multi(tester.hyps, tester.refer)
        else:
            scores_all = utils.pyrouge_score_all(tester.hyps, tester.refer)
    else:
        rouge = Rouge()
        scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)

    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
        scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
          + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
          + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
              scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(res)

    tester.getMetric()
    tester.SaveDecodeFile()
    logger.info('[INFO] End of test | time: {:5.2f}s | test loss {:5.4f} | '.format((time.time() - iter_start_time),
                                                                                    float(running_avg_loss)))


def main():
    args = pars_args()
    parser = argparse.ArgumentParser(description='ConGNN-SUM Model')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    hps = args
    hps = set_device(hps=hps)

    torch.set_printoptions(threshold=50000)

    # File paths
    DATA_FILE = os.path.join(args.data_dir, "test.label.jsonl")
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.txt")
    LOG_PATH = args.log_root

    # train_log setting
    if not os.path.exists(LOG_PATH):
        logger.exception("[Error] Logdir %s doesn't exist. Run in train mode to create it.", LOG_PATH)
        raise Exception("[Error] Logdir %s doesn't exist. Run in train mode to create it." % (LOG_PATH))
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "test_" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
    vocab = Vocab(VOCAL_FILE, args.vocab_size)
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim)
    if args.word_embedding:
        embed_loader = Word_Embedding(args.embedding_path, vocab)
        vectors = embed_loader.load_my_vecs(args.word_emb_dim)
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        embed.weight.requires_grad = args.embed_train

    logger.info(hps)

    test_w2s_path = os.path.join(args.cache_dir, "test.w2s.tfidf.jsonl")
    if hps.model == "HSG":
        # model = HSumGraph(hps, embed)
        model = Model(hps, embed)
        logger.info("[MODEL] ConGNN-SUM ")
        loader = data_loaders.make_dataloader(
            data_file=DATA_FILE, vocab=vocab, hps=hps, filter_word=FILTER_WORD, w2s_path=test_w2s_path,
            graphs_dir=os.path.join(args.cache_dir, "graphs\\test")
        )
        if hps.fill_graph_cache:
            return

        #
        # dataset = SummarizationDataSet(data_path=DATA_FILE, vocab=vocab, hps=hps, filter_word_path=FILTER_WORD,
        #                                w2s_path=test_w2s_path, graphs_dir=None)

        # loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=False, num_workers=2,
        #                                      collate_fn=graph_collate_fn)
    # elif hps.model == "HDSG":
    #     model = HSumDocGraph(hps, embed)
    #     logger.info("[MODEL] HeterDocSumGraph ")
    #     test_w2d_path = os.path.join(args.cache_dir, "test.w2d.tfidf.jsonl")
    #     dataset = MultiSummarizationDataSet(DATA_FILE, vocab, hps.doc_max_timesteps, hps.sent_max_len, FILTER_WORD, test_w2s_path,
    #                                         test_w2d_path)
    #     loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=2,
    #                                          collate_fn=graph_collate_fn)
    else:
        logger.error("[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")

    if args.cuda:
        hps.device = torch.device("cuda:0")
        logger.info("[INFO] Use cuda")

    else:
        hps.device = torch.device("cuda:0")
        logger.info("[INFO] Use CPU")

    logger.info("[INFO] Decoding...")
    if hps.test_model == "multi":
        for i in range(3):
            model_name = "evalbestmodel_%d" % i
            run_test(model, loader.dataset, loader, model_name, hps)
    else:
        print(model)
        run_test(model, loader.dataset, loader, hps.test_model, hps)


if __name__ == '__main__':
    main()
