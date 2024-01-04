import os
import shutil
import time
import numpy as np
import torch
import dgl
from config import _DEBUG_FLAG_
from data_manager import data_loaders
from tools.logger import *
from tools.utils import save_model
from runner.evaluation import run_eval


def setup_training(model, hps, data_variables):
    train_dir = os.path.join(hps.save_root, "train")
    if os.path.exists(train_dir) and hps.restore_model != 'None':
        logger.info("[INFO] Restoring %s for training...", hps.restore_model)
        # best_model_file = os.path.join(train_dir, hps.restore_model)
        # model.load_state_dict(torch.load(best_model_file))
        model.HSG.load_state_dict(torch.load(hps.restore_model))
        # model.load_state_dict(torch.load(hps.restore_model))
        # hps.save_root = hps.save_root + "_reload"
    else:
        logger.info("[INFO] Create new model for training...")
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir)

    try:
        run_training(model, hps, data_variables=data_variables)
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"))


class Trainer:
    def __init__(self, model, hps, train_dir):
        self.model = model
        self.hps = hps
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hps.lr)
        #
        # self.optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, self.model.parameters()), lr=hps.lr)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.best_train_loss = None
        self.best_loss = None
        self.best_F = None
        self.non_descent_cnt = 0
        self.saveNo = 0
        self.epoch = 1
        self.epoch_avg_loss = 0
        self.train_dir = train_dir
        self.report_epoch = 100

    def run_epoch(self, train_loader):
        epoch_start_time = time.time()

        train_loss = 0.0
        epoch_loss = 0.0
        iters_start_time = time.time()
        iter_start_time = time.time()
        for i, (G, index) in enumerate(train_loader):
            loss = self.train_batch(G=G)
            # print(f"{i}=>{loss}")
            train_loss += float(loss.data)
            epoch_loss += float(loss.data)
            if i % self.report_epoch == self.report_epoch - 1:
                if _DEBUG_FLAG_:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            logger.debug(name)
                            logger.debug(param.grad.data.sum())
                batch_time_sum = time.time() - iters_start_time
                iters_start_time = time.time()
                logger.info('| end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '.format(i, (
                        batch_time_sum / self.report_epoch), float(train_loss / self.report_epoch)))
                train_loss = 0.0
                self.save_current_model()
            iter_start_time = time.time()

        # self.save_epoch_model()

        self.epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info(' | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '.format(self.epoch, (
                time.time() - epoch_start_time), float(self.epoch_avg_loss)))
        return epoch_loss

    def train_batch(self, G):
        G = G.to(self.hps.device)  # TODO i think G is in device
        outputs = self.model.forward(G)  # [n_snodes, 2]
        snode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        label = G.ndata["label"][snode_id].sum(-1)  # [n_nodes]
        G.nodes[snode_id].data["loss"] = self.criterion(outputs, label.to(self.hps.device)).unsqueeze(
            -1)  # [n_nodes, 1]
        loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]
        loss = loss.mean()
        if not (np.isfinite(loss.data.cpu())).numpy():
            logger.error("train Loss is not finite. Stopping.")
            logger.info(loss)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    logger.info(name)
                    # logger.info(param.grad.data.sum())
            raise Exception("train Loss is not finite. Stopping.")
        self.optimizer.zero_grad()
        loss.backward()
        if self.hps.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hps.max_grad_norm)

        self.optimizer.step()
        return loss

    def change_learning_rate(self):
        if self.hps.lr_descent:
            new_lr = max(5e-6, self.hps.lr / (self.epoch + 1))
            for param_group in list(self.optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)

    def save_epoch_model(self):
        if not self.best_train_loss or self.epoch_avg_loss < self.best_train_loss:
            save_file = os.path.join(self.train_dir, "bestmodel")
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s',
                        float(self.epoch_avg_loss),
                        save_file)
            save_model(self.model, save_file)
            self.best_train_loss = self.epoch_avg_loss
        elif self.epoch_avg_loss >= self.best_train_loss:
            logger.error("[Error] training loss does not descent. Stopping supervisor...")
            save_model(self.model, os.path.join(self.train_dir, "earlystop"))
            sys.exit(1)

    def save_current_model(self):
        save_file = os.path.join(self.train_dir, "current")
        save_model(self.model, save_file)


def run_training(model, hps, data_variables):
    trainer = Trainer(model=model, hps=hps, train_dir=os.path.join(hps.save_root, "train"))
    train_size = 287000
    n_part = 16

    print(f"data_loader")

    for epoch in range(1, hps.n_epochs + 1):
        logger.info(f"train started in epoch={epoch}")

        logger.info("train loader read")

        trainer.epoch = epoch
        model.train()

        for train_data_part in range(n_part + 1):
            if train_data_part == n_part:
                from_index = train_data_part * train_size // n_part
                to_index = None
            else:
                from_index = train_data_part * train_size // n_part
                to_index = (train_data_part + 1) * train_size // n_part
            train_loader = data_loaders.make_dataloader(data_file=data_variables["train_file"],
                                                        vocab=data_variables["vocab"], hps=hps,
                                                        filter_word=data_variables["filter_word"],
                                                        w2s_path=data_variables["train_w2s_path"],
                                                        graphs_dir=os.path.join(data_variables["graphs_dir"],
                                                                                "train"),
                                                        from_index=from_index,
                                                        to_index=to_index,
                                                        shuffle=True
                                                        )

            print(f"train loader from {from_index} to {to_index} started epoch started ")

            trainer.run_epoch(train_loader=train_loader)
            print(f"train loader from {from_index} to {to_index} started epoch finished ")
            del train_loader

        valid_loader = data_loaders.make_dataloader(data_file=data_variables["valid_file"],
                                                    vocab=data_variables["vocab"], hps=hps,
                                                    filter_word=data_variables["filter_word"],
                                                    w2s_path=data_variables["val_w2s_path"],
                                                    graphs_dir=os.path.join(data_variables["graphs_dir"],
                                                                            "val"))

        best_loss, best_F, non_descent_cnt, saveNo = run_eval(model, valid_loader, valid_loader.dataset, hps,
                                                              trainer.best_loss,
                                                              trainer.best_F, trainer.non_descent_cnt,
                                                              trainer.saveNo)

        del valid_loader

        if non_descent_cnt >= 3:
            logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
            save_model(model, os.path.join(data_variables["train_dir"], "earlystop"))
            return
