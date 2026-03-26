# -------------------------------------------------------
# @Author: vegesin
# @Date: 2025-12-18
# @FilePath: \radar_pc\src\main\utils\Train.py
# @Description: 框架的训练相关脚本
# -------------------------------------------------------

import os
from random import sample
import sys
import csv
import datetime

# from drjit import device
# from pytorch_lightning import data_loader
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# from spikingjelly.clock_driven import functional # 新版本存在类检查

from .build import *
from .base import *
from .metrics import *

# logger配置
log: myLogger = logging.getLogger("mylogger")


class Trainer:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # 1. 组件加载
        self.net = build_model(cfg).to(self.device)
        self.optimizer = build_optimizer(cfg, self.net)
        self.scheduler = build_scheduler(cfg, self.optimizer)
        self.loss_func = build_loss_func(cfg).to(self.device)

        # 2. 数据加载器配置，一次性获取所有需要的 Loaders
        self.loaders = build_loaders(cfg)

        # 3. 路径初始化
        self.save_dir = build_save_path(cfg)
        self.model_save_path = os.path.join(self.save_dir, "checkpoints", "best.pth")

        # 4.全局参数
        self.cfar_th = None

    # >--------------------------------------------------------------
    # >训练测试过程种需要的基础功能函数 （模型保存、恒虚警门限）
    # >--------------------------------------------------------------
    def save_ckpt(self, epoch: int, is_best=False):
        ''' 检查点和最优模型保存'''
        ckpt_save_dir = os.path.join(self.save_dir, "checkpoints")
        assert os.path.isdir(ckpt_save_dir), f"err: 找不到检查点目录 {ckpt_save_dir}，请手动创建后再运行。"

        # yapf:disable
        model_save_dict = {
            'epoch': epoch,
            'model_state': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'cfg': vars(self.cfg)
            }
        # yapf:enable

        # 最优模型保存，保存之后退出
        if is_best:
            best_save_name = f"best_ckpt.pth"
            best_save_path = os.path.join(ckpt_save_dir, best_save_name)
            torch.save(model_save_dict, best_save_path)
            log.note(f"最优模型保存保存到{best_save_path}")
            return

        # 检查点保存
        ckpt_save_name = f"ckpt_{epoch}.pth"
        ckpt_save_path = os.path.join(ckpt_save_dir, ckpt_save_name)
        torch.save(model_save_dict, ckpt_save_path)
        log.info(f"当前检查点保存{ckpt_save_path}")

    def get_cfar_th(self, pfa: float = 1e-3, loader: torch.utils.data.DataLoader = None, num_steps: int = None, f_name: str = None):
        """
        ### TODO : 当前的虚警门限获取函数为了兼容[B,2]输出存在冗余，'z1'字段参数
        
        恒虚警门限计算
        1. 核心目标：根据指定的虚警概率(Pfa)计算神经网络输出的检测门限值
        2. 计算原理：
           - 使用纯杂波样本（标签为0）的神经网络输出建立杂波分布模型
           - 对杂波样本的输出值进行排序，找到对应(1-Pfa)分位数的值作为门限
           - 当检测时，输出值大于此门限则判为目标，小于等于则判为杂波
        
        Args:
            pfa (float): 目标虚警率 (如 1e-3)
            loader (DataLoader): 包含纯杂波样本或训练数据集的加载器
        
        Returns:
            threshold (float): 计算得到的恒虚警决策门限
        """
        self.net.eval()
        clutter_softmax, target_scores = [], []
        clutter_logits, target_logits = [], []
        clutter_delta_logits, target_delta_logits = [], []

        log.info(f"正在计算 Pfa={pfa} 对应的门限阈值...")

        with torch.no_grad():
            pbar = tqdm(loader, desc="控制恒虚警,计算门限", ncols=80)
            for signals, labels, *rest in pbar:
                inputs = signals.to(self.device)
                labels = labels.to(self.device)

                # 这里计算三种分布 ： z1-z0 | z1 |softmax
                outputs = self.net(inputs) # 现在这里维度为[B,1] | [B,2]

                if outputs.size(1) == 2:
                    probs = F.softmax(outputs, dim=1)[:, 1]
                    logits_chan0 = outputs[:, 0]
                    logits_chan1 = outputs[:, 1]
                    delta_logits = logits_chan1 - logits_chan0
                elif outputs.size(1) == 1:
                    # 单通道输出，兼容框架 | 当单通道输出的时候，下面这个值没有使用
                    logits = outputs.squeeze(1) # [B]
                    probs = F.sigmoid(logits)   # 等价 soft score

                    # 为了兼容框架
                    delta_logits = logits
                    logits_chan1 = logits

                # 筛选出真实的杂波样本
                clutter_mask = (labels == 0)
                target_mask = (labels == 1)

                if clutter_mask.any():
                    clutter_softmax.extend(probs[clutter_mask].cpu().tolist())
                    clutter_logits.extend(logits_chan1[clutter_mask].cpu().tolist())
                    clutter_delta_logits.extend(delta_logits[clutter_mask].cpu().tolist())
                if target_mask.any():
                    target_scores.extend(probs[target_mask].cpu().tolist())
                    target_logits.extend(logits_chan1[target_mask].cpu().tolist())
                    target_delta_logits.extend(delta_logits[target_mask].cpu().tolist())

        if not clutter_softmax:
            log.error("未在数据集中发现杂波样本，无法计算门限,返回默认门限0.5")
            return 0.5

        # 排序并根据 Pfa 确定分位数
        clutter_softmax.sort() # 升序重排 小->大 1 - pfa 尾部
        num_clutter = len(clutter_softmax)

        idx = int((1 - pfa) * num_clutter)
        idx = max(0, min(idx, num_clutter - 1))

        # softmax 门限
        th_softmax = clutter_softmax[idx]

        # logit1 门限
        clutter_logits_sorted = sorted(clutter_logits)
        th_logit1 = clutter_logits_sorted[idx]

        # delta logit 门限
        clutter_delta_logits_sorted = sorted(clutter_delta_logits)
        th_delta = clutter_delta_logits_sorted[idx]

        if f_name is not None:

            # f_name = match_npynum(f_name)

            # # softmax(z1)
            # softmax_name = f_name + '_softmax(z1)'
            # # 这里绘图不显示应该是 [B,2]的问题
            # plot_score_distribution(clutter_softmax, target_scores, th_softmax, pfa, softmax_name, self.save_dir)

            # z1
            z1_name = f_name + '_z1'
            plot_logit_distribution(clutter_logits, target_logits, th_logit1, pfa, z1_name, self.save_dir)

            # # z1-z0
            # delta_name = f_name + '_z1-z0'
            # plot_logit_distribution(clutter_delta_logits, target_delta_logits, th_delta, pfa, delta_name, self.save_dir)

            log.info(f"分布图已存入{self.save_dir}")

        # 返回不同的判决门限
        match self.cfg.logits_process:
            case "z1_z0":
                threshold = th_delta
            case "z1":
                threshold = th_logit1
            case "softmax":
                threshold = th_softmax
            case _:
                raise ValueError(f"Unsupported score mode: {self.cfg.logits_process}")

        log.note(f"门限计算完成: 总杂波样本数={num_clutter}, 输出处理模式{self.cfg.logits_process},计算门限={threshold:.6f}")

        return threshold

    # >--------------------------------------------------------------
    # >通用神经网络 训练验证相关函数 [Batchsize,shape]
    # >--------------------------------------------------------------
    def run_train(self):
        '''
        经典模型训练函数
        当前函数使用train_val loader模式，加载loader字典中返回的键固定为'trian' , 'val'
        '''

        num_epochs = self.cfg.num_epochs
        train_loader = self.loaders.get('train')
        val_loader = self.loaders.get('val')

        best_loss = 1e9

        for epoch in range(1, num_epochs + 1):

            avg_train_loss = self._train_one_epoch(epoch, train_loader)

            # --- Validation Loop ---
            if val_loader:

                val_results = self._val_one_epoch(loader=val_loader, metrics_func=None)

                # 获取loss
                current_val_loss = val_results.get("loss", 0)

                log.info(f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {current_val_loss:.4f} ")

                # 最优模型保存 验证集上最小loss保存
                if current_val_loss < best_loss:
                    best_loss = current_val_loss
                    self.save_ckpt(epoch=epoch, is_best=True)

            # 周期检查点保存
            if epoch % self.cfg.model_save_interval == 0:
                self.save_ckpt(epoch=epoch, is_best=False)

            if self.scheduler is not None:
                self.scheduler.step()

    def _train_one_epoch(self, epoch: int, loader: torch.utils.data.DataLoader) -> float:
        """
        内部函数,执行一个epoch的训练.
        
        Args:
            epoch (int): 当前轮次
            loader (torch.utils.data.DataLoader): 从init中创建的loaders获取train_loader,由调用这个内部训练函数的模式传入

        Returns:
           avg_loss(float) : 当前轮次的平均损失
        """
        self.net.train()
        total_loss = 0.0
        batch_count = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", ncols=80)
        for signals, labels, sample_str, *rest in pbar:

            inputs, labels = signals.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            if outputs.size(1) == 1:
                outputs = outputs.squeeze(1)

            loss = self.loss_func(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 当前epoch平均loss
        avg_loss = total_loss / batch_count

        return avg_loss

    def _val_one_epoch(self, loader: torch.utils.data.DataLoader, metrics_func=None):
        """内部函数,执行一个epoch的验证.

        Args:
            loader (torch.utils.data.DataLoader): 从init中创建的loaders获取val_loader,由调用这个内部训练函数的模式传入
            metrics_func (_type_, optional): 其他复杂指标的计算逻辑传入,这个指标计算函数从调用模式里面传入.

        Returns:
            results(dict) : 验证指标字典,没有传入metrics_func的时候,默认只有results["loss"]验证集的loss
        """
        self.net.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for signals, labels, *rest in loader:
                inputs, labels = signals.to(self.device), labels.to(self.device)

                outputs = self.net(inputs)

                # 1. 计算损失 [b,1] --> [b]
                if outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)

                loss = self.loss_func(outputs, labels)
                val_loss += loss.item()

                if metrics_func is not None:
                    # 这里先不考虑兼容 [b,1] [b,2]
                    preds = (outputs >= self.cfar_th).long()

                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

        # 指标
        results = {"loss": val_loss / len(loader)}

        if metrics_func is not None and len(all_preds) > 0:
            y_pred = torch.cat(all_preds).numpy()
            y_true = torch.cat(all_labels).numpy()

            # 执行metrics_func引用，进行相关指标计算
            extra_metrics = metrics_func(y_true, y_pred)
            results.update(extra_metrics)

        return results

    # >--------------------------------------------------------------
    # >推理评估相关函数
    # >--------------------------------------------------------------

    # 加载模型
    def load_model(self, resume=False):
        '''
        模型推理评估下,加载保存的ckpt
        
        # TODO :param resume: 从检查节点加载模型重新训练.
        '''

        load_model_path = self.cfg.load_mode_path

        checkpoint = torch.load(load_model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state'])

        log.note(f"载入模型{load_model_path}")

    def profile_model(self):
        """
        # TODO :  ✔ 1.兼容封装到 Tranier类中
        #           实现模型参数量/计算量统计
        
        ### @ 统计模型的参数量计算量
       
        
        返回:
            dict: 包含评估结果的字典。缺失的属性（如 CNN 的 SOPs）将赋值为 None。
        """
        model = self.net
        device = self.device
        dataloader = self.loaders['test'] # 这里的dataloader使用['test'] 固定键名

        model.eval()
        model.to(device)

        # 统计总参数量
        total_params = sum(p.numel() for p in model.parameters())


        pass

    def run_test(self):
        """
        测试模式：遍历所有测试加载器，计算指标并保存到 CSV
        
        # fix 匹配训练集和测试集
        当前的build loader传出的loader字典里面，获取对应的字典需要 val / test / train 这种前置键值匹配
        当模式不兼容时候，这里会报错
        后续还是设计一次只载入train test val 三个loader， 多个数据集评估通过bash脚本给启动py传递参数实现。
        
        """
        log.info("开始测试模式...")

        self.load_model()

        #  self.loaders 是一个包含多个测试集的字典
        all_loader_dict = self.loaders

        metrics_func = metrics_binary_class # TODO: 这个参数配置封装到cfg

        all_loader_key_list = list(all_loader_dict.keys())

        train_keys = match_dictkey(all_loader_key_list, npy_prefix='train')
        test_keys = match_dictkey(all_loader_key_list, npy_prefix='test') # 按照字段匹配，开始的没划分test 使用test字段不能匹配出测试集

        # train_loaders_dict = {key: all_loader_dict[key] for key in train_keys if key in all_loader_dict}
        test_loaders_dict = {key: all_loader_dict[key] for key in test_keys if key in all_loader_dict}

        data2csv = []

        # 遍历测试集
        for f_name, test_loader in test_loaders_dict.items():
            log.info(f"---------------------测试: {f_name}---------------------")

            # 根据测试集f_name匹配相应的训练集，获取虚警门限
            npy_num = match_npynum(f_name)

            # 匹配对应的训练集
            # 返回长度为1的list，使用,解包
            train_loader_key, = match_dictkey(all_loader_key_list, npy_prefix='train', npy_number=npy_num)

            if train_loader_key is None or npy_num is None:
                log.warn(f"当前的{f_name}没有对应的训练集计算虚警门限")

            # 使用训练数据集计算虚警门限
            self.cfar_th = self.get_cfar_th(pfa=self.cfg.pfa, loader=all_loader_dict[train_loader_key], f_name=f_name)

            # 调用通用的验证函数
            val_results = self._val_one_epoch(loader=test_loader, metrics_func=metrics_func)
            val_results["th"] = self.cfar_th

            # 提取基础指标用于日志打印
            acc = val_results.get("acc", 0.0)
            pd = val_results.get("pd", 0.0)
            loss = val_results.get("loss", 0.0)
            pfa = val_results.get("pfa", 0.0)

            # --- 混淆矩阵 ---
            tn = val_results.get("tn", 0)
            fp = val_results.get("fp", 0)
            fn = val_results.get("fn", 0)
            tp = val_results.get("tp", 0)

            prinit_confusion_matrix(f_name, tn, fp, fn, tp) # 混淆矩阵格式化打印
            log.note(f"[{f_name}] -> Pfa: {pfa:.6f} | Pd: {pd*100:.2f}% \n")

            # CSV 数据行：将数据集名称和所有指标合并
            row = {"dataset_name": f_name}
            row.update(val_results)

            data2csv.append(row)

        # save
        save_dir = self.save_dir
        csv_path = os.path.join(save_dir, f'metrics_pfa{self.cfg.pfa}.csv')
        # 获取所有字段名（从第一个字典的键）
        fieldnames = data2csv[0].keys()

        # 写入 CSV 文件
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data2csv)

        log.info(f"测试结果已保存到: {csv_path}")
        log.info(f"共保存了 {len(data2csv)} 个数据集的测试结果")


if __name__ == "__main__":

    log.info("针对Train的单独测试")

    pass
