# -------------------------------------------------------
# @Author: vegesin
# @Date: 2025-12-18
# @FilePath: \SNN\src\main\train.py
# @Description: 训练、推理测试类
# TODO ： 后续看如何设计这个trainer 适配多任务场景 （二分类，多分类，分割等）
# -------------------------------------------------------

import csv
import json
import os
from typing import Optional

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, neuron
from tqdm import tqdm

from .logger import log
from .base import DataConfig, EvalConfig, ExperimentConfig, ModelConfig, RunConfig, TrainConfig
from .data_factory import build_data_components
from .metrics import metrics_binary_class, plot_logit_distribution, print_confusion_matrix
from .model_factory import build_model
from .train_factory import build_loss_func, build_optimizer, build_save_path, build_scheduler
from src.utils import save_dict_as_json


class Trainer:
    """Unified execution class for train/test/profile modes.

    Args:
        experiment: 结构化实验配置。

    Attributes:
        runtime: 运行时配置。
        model_cfg: 模型配置。
        data_cfg: 数据配置。
        train_cfg: 训练配置。
        eval_cfg: 测试与评分配置。
    """

    def __init__(self, experiment: ExperimentConfig):
        """Initialize trainer state and all runtime dependencies.

        Args:
            experiment: Structured experiment configuration.
        """
        # self.config_snapshot = experiment.to_dict() # 这里面可能包含一些实例化之后的类，函数引用等等

        self.run_cfg: RunConfig = experiment.run
        self.model_cfg: ModelConfig = experiment.model
        self.data_cfg: DataConfig = experiment.data
        self.train_cfg: TrainConfig = experiment.train
        self.eval_cfg: EvalConfig = experiment.eval

        run_cfg = self.run_cfg
        model_cfg = self.model_cfg
        data_cfg = self.data_cfg
        train_cfg = self.train_cfg

        # $ 传入上面的配置类之后，在下面的init中统一初始化好需要的组件

        self.save_dir = build_save_path(run_cfg.save_dir, allow_overwrite=run_cfg.allow_overwrite)
        self.device = torch.device(run_cfg.device if torch.cuda.is_available() else "cpu")

        model_extra = model_cfg.get_extra()
        self.net = build_model(
            model_name=model_cfg.model_name,
            model_extra=model_extra,
        ).to(self.device)

        self.optimizer = build_optimizer(
            model=self.net,
            optimizer_type=train_cfg.optimizer_type,
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

        self.scheduler = build_scheduler(
            optimizer=self.optimizer,
            scheduler_type=train_cfg.scheduler_type,
            num_epochs=train_cfg.num_epochs,
            min_lr=train_cfg.min_lr,
        )

        self.loss_func = build_loss_func(train_cfg.loss_func).to(self.device)

        self.loaders = build_data_components(data_cfg=data_cfg)

        self.model_save_path = os.path.join(self.save_dir, "checkpoints", "best.pth")
        self.cfar_th: Optional[float] = None

    def save_ckpt(self, epoch: int, is_best: bool = False):
        """Save periodic or best checkpoint.

        Args:
            epoch: Current epoch number.
            is_best: Whether this checkpoint is the best checkpoint.
        """
        ckpt_save_dir = os.path.join(self.save_dir, "checkpoints")
        assert os.path.isdir(ckpt_save_dir), f"checkpoint directory does not exist: {ckpt_save_dir}"

        model_save_dict = {
            "epoch": epoch,
            "model_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }

        if is_best:
            best_save_path = os.path.join(ckpt_save_dir, "best_ckpt.pth")
            torch.save(model_save_dict, best_save_path)
            log.note(f"Saved best checkpoint to {best_save_path}")
            return

        ckpt_save_path = os.path.join(ckpt_save_dir, f"ckpt_{epoch}.pth")
        torch.save(model_save_dict, ckpt_save_path)
        log.info(f"Saved checkpoint to {ckpt_save_path}")

    def get_cfar_th(self, pfa: float, loader: torch.utils.data.DataLoader, f_name: str | None = None):
        """Estimate CFAR threshold from clutter-dominant samples.

        中文说明：
            测试阶段的二分类判决不直接写死阈值，而是先根据给定的
            `pfa` 在辅助数据集上统计门限。对于双输出模型，会同时统计：
            `softmax(z1)`、`z1`、`z1-z0` 三种候选分数，再根据
            `logits_process` 选择最终门限。

        Args:
            pfa: Target false alarm probability.
            loader: Dataloader used for threshold estimation.
            f_name: Optional experiment/display name for plots.

        Returns:
            float: Estimated threshold.
        """
        self.net.eval()

        clutter_scores = []
        target_scores = []

        log.info(f"Computing CFAR threshold with pfa={pfa}")

        with torch.no_grad():
            pbar = tqdm(loader, desc="Compute threshold", ncols=80)
            for signals, labels, *rest in pbar:
                inputs = signals.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(inputs)
                outputs = outputs.view(-1)
                labels = labels.view(-1)

                clutter_mask = labels == 0
                target_mask = labels == 1

                if clutter_mask.any():
                    clutter_scores.extend(outputs[clutter_mask].cpu().tolist())
                if target_mask.any():
                    target_scores.extend(outputs[target_mask].cpu().tolist())

        if not clutter_scores:
            log.error("No clutter samples found while computing CFAR threshold, fallback to 0.5")
            return 0.5

        num_clutter = len(clutter_scores)
        idx = int((1 - pfa) * num_clutter)
        idx = max(0, min(idx, num_clutter - 1))
        threshold = sorted(clutter_scores)[idx]

        if f_name is not None:
            plot_logit_distribution(clutter_scores, target_scores, threshold, pfa, f_name, self.save_dir)

        log.note(f"CFAR threshold ready: samples={num_clutter}, threshold={threshold:.6f}")
        return threshold

    def run_train(self):
        """Run the standard training loop.

        当前逻辑优先使用 `val` loader 做验证；
        如果没有显式验证集，则回退到 `test` loader。
        """
        num_epochs = self.train_cfg.num_epochs
        run_cfg = self.run_cfg
        model_save_interval = run_cfg.model_save_interval
        train_loader = self.loaders.get("train")
        val_loader = self.loaders.get("val") or self.loaders.get("test")

        if train_loader is None:
            raise ValueError("train loader is required for run_train")

        best_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            avg_train_loss = self._train_one_epoch(epoch, train_loader)

            if val_loader is not None:
                val_results = self._val_one_epoch(loader=val_loader, metrics_func=None)
                current_val_loss = val_results.get("loss", 0.0)
                log.info(f"Epoch [{epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {current_val_loss:.4f}")

                if current_val_loss < best_loss:
                    best_loss = current_val_loss
                    self.save_ckpt(epoch=epoch, is_best=True)

            if epoch % model_save_interval == 0:
                self.save_ckpt(epoch=epoch, is_best=False)

            if self.scheduler is not None:
                self.scheduler.step()

    def _train_one_epoch(self, epoch: int, loader: torch.utils.data.DataLoader) -> float:
        """Train the model for one epoch.

        Args:
            epoch: Current epoch number.
            loader: Train dataloader.

        Returns:
            float: Average training loss for the epoch.
        """
        self.net.train()
        total_loss = 0.0
        batch_count = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", ncols=80)
        for signals, labels, sample_str, *rest in pbar:

            inputs = signals.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            loss = self.loss_func(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(batch_count, 1)

    def _val_one_epoch(self, loader: torch.utils.data.DataLoader, metrics_func=None):
        """Run one validation or test pass.

        Args:
            loader: Validation/test dataloader.
            metrics_func: Optional metric callable. When omitted, only loss is
                computed.

        Returns:
            dict: Loss and optional metric values.
        """
        self.net.eval()
        val_loss = 0.0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for signals, labels, *rest in loader:
                inputs = signals.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(inputs)
                loss = self.loss_func(outputs, labels)
                val_loss += loss.item()

                if metrics_func is not None:
                    all_outputs.append(outputs.detach().cpu())
                    all_labels.append(labels.cpu())

        results = {"loss": val_loss / max(len(loader), 1)}

        if metrics_func is not None and all_outputs:
            outputs = torch.cat(all_outputs)
            labels = torch.cat(all_labels)
            results.update(metrics_func(outputs, labels))

        return results

    def load_model(self, resume: bool = False):
        """Load model weights from the configured checkpoint path.

        Args:
            resume: Reserved flag for future resume-training support.
        """
        del resume

        run_cfg = self.run_cfg
        load_model_path = run_cfg.load_model_path
        checkpoint = torch.load(load_model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state"])
        log.note(f"Loaded model from {load_model_path}")

    def profile_model(self):
        """Profile model parameters and approximate compute cost.

        中文说明：
            这里除了统计 Params/MACs，还会在 SNN 场景下基于脉冲稀疏性
            估算 SOPs 和平均发放率。对于特定 attention 模块，还补充了
            一部分近似计算开销。

        Returns:
            dict: Profiling summary dictionary.
        """
        model = self.net
        device = self.device
        dataloader = self.loaders.get("test")

        if dataloader is None:
            raise ValueError("test loader is required for profiling")

        self.load_model()
        model.eval()
        model.to(device)

        total_params = sum(param.numel() for param in model.parameters())
        stats = {"total_macs": 0.0, "total_sops": 0.0, "neuron_stats": {}, "samples_count": 0}
        hooks = []
        is_snn = False

        def neuron_hook(module, args, output):
            """Collect spike statistics for neuron-like modules."""
            if module not in stats["neuron_stats"]:
                stats["neuron_stats"][module] = {"spikes": 0.0, "elements": 0.0}

            out_tensor = output[0] if isinstance(output, tuple) else output
            spikes = (out_tensor > 0).float().sum().item()
            elements = out_tensor.numel()
            stats["neuron_stats"][module]["spikes"] += spikes
            stats["neuron_stats"][module]["elements"] += elements

        def compute_cost_hook(module, args, output):
            """Accumulate dense MACs or spike-driven SOPs for supported layers."""
            x = args[0]
            if isinstance(x, tuple):
                x = x[0]

            out_tensor = output[0] if isinstance(output, tuple) else output
            is_spike_input = len(x.unique()) <= 2 if x.numel() > 0 else False

            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                kernel_product = 1
                for kernel_size in module.kernel_size:
                    kernel_product *= kernel_size
                total_dense_macs = out_tensor.numel() * (module.in_channels // module.groups) * kernel_product
            elif isinstance(module, nn.Linear):
                total_dense_macs = out_tensor.numel() * module.in_features
            else:
                return

            if is_spike_input:
                sparsity = (x > 0).float().mean().item()
                stats["total_sops"] += total_dense_macs * sparsity
            else:
                stats["total_macs"] += total_dense_macs

        for name, module in model.named_modules():
            del name

            module_name = module.__class__.__name__
            is_snn_node = isinstance(module, neuron.BaseNode) or "LIFNode" in module_name or "IFNode" in module_name

            if is_snn_node:
                hooks.append(module.register_forward_hook(neuron_hook))
                is_snn = True
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                hooks.append(module.register_forward_hook(compute_cost_hook))

        log.info("Profiling model...")
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Profile", ncols=80)
            for inputs, targets, *rest in pbar:
                del targets, rest

                inputs = inputs.to(device)
                stats["samples_count"] += inputs.size(1) if inputs.dim() == 5 else inputs.size(0)

                _ = model(inputs)

                if is_snn:
                    functional.reset_net(model)

        for hook in hooks:
            hook.remove()

        if is_snn:
            for name, module in model.named_modules():
                del name

                module_name = module.__class__.__name__
                if module_name in ["MS_SSA_Conv", "myMS_SSA_Conv"]:
                    try:
                        q_stats = stats["neuron_stats"].get(module.q_lif)
                        k_stats = stats["neuron_stats"].get(module.k_lif)
                        if q_stats and k_stats and q_stats["elements"] > 0:
                            elements = q_stats["elements"]
                            firing_rate_q = q_stats["spikes"] / q_stats["elements"]
                            firing_rate_k = k_stats["spikes"] / k_stats["elements"]

                            if module_name == "MS_SSA_Conv":
                                stats["total_sops"] += elements * firing_rate_k
                            else:
                                stats["total_sops"] += 4 * elements * (firing_rate_q + firing_rate_k)
                    except AttributeError:
                        pass
                elif module_name == "DSSA":
                    try:
                        input_stats = stats["neuron_stats"].get(module.activation_in)
                        attn_stats = stats["neuron_stats"].get(module.activation_attn)
                        if input_stats and attn_stats and input_stats["elements"] > 0:
                            elements = input_stats["elements"]
                            firing_rate_x = input_stats["spikes"] / input_stats["elements"]
                            firing_rate_attn = attn_stats["spikes"] / attn_stats["elements"]
                            stats["total_sops"] += elements * module.lenth * (firing_rate_x + firing_rate_attn)
                    except AttributeError:
                        pass

        samples = stats["samples_count"] if stats["samples_count"] > 0 else 1
        avg_macs = stats["total_macs"] / samples

        results = {
            "total_params": total_params,
            "macs_per_sample": avg_macs,
            "sops_per_sample": None,
            "average_firing_rate": None,
            "neuron_firing_rates": None,
        }

        average_firing_rate = 0.0
        if is_snn:
            results["sops_per_sample"] = stats["total_sops"] / samples

            firing_rates = {}
            for idx, (module, module_stats) in enumerate(stats["neuron_stats"].items()):
                firing_rate = module_stats["spikes"] / module_stats["elements"] if module_stats["elements"] > 0 else 0
                firing_rates[f"{module.__class__.__name__}_{idx}"] = round(firing_rate, 6)
            results["neuron_firing_rates"] = firing_rates

            if firing_rates:
                average_firing_rate = sum(firing_rates.values()) / len(firing_rates)
                results["average_firing_rate"] = round(average_firing_rate, 6)

        log.info("\n" + "-" * 40)
        log.info("Profile summary:")
        log.note(f" -> Params: {results['total_params'] / 1e6:.4f} M")
        log.note(f" -> MACs  : {results['macs_per_sample'] / 1e6:.4f} M")

        if is_snn:
            log.note(f" -> SOPs  : {results['sops_per_sample'] / 1e6:.4f} M")
            if results["average_firing_rate"] is not None:
                log.note(f" -> FR    : {results['average_firing_rate']:.2%}")

        log.info("-" * 40 + "\n")

        save_dir = os.path.join(self.save_dir, "inference")
        csv_path = os.path.join(save_dir, "params_macs_sops.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Params (M)", f"{results['total_params'] / 1e6:.6f}"])
            writer.writerow(["MACs per sample (M)", f"{results['macs_per_sample'] / 1e6:.6f}"])
            if is_snn:
                writer.writerow(["SOPs per sample (M)", f"{results['sops_per_sample'] / 1e6:.6f}"])
                writer.writerow(["Average Firing Rate", f"{average_firing_rate:.4%}"])

        json_path = os.path.join(save_dir, "profile_details.json")
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        log.info(f"Saved profile CSV: {csv_path}")
        log.info(f"Saved profile JSON: {json_path}")
        return results

    def _get_metrics_func(self):
        """Get the configured metric function."""
        metrics_func = self.eval_cfg.metrics_func or metrics_binary_class
        if not callable(metrics_func):
            log.error(f"metrics_func must be callable, got {type(metrics_func).__name__}: {metrics_func}")
        assert callable(metrics_func), f"metrics_func must be callable, got {type(metrics_func).__name__}: {metrics_func}"
        return metrics_func

    def run_base_test(self):
        """Run a task-agnostic test pass on the test loader."""
        log.info("Running base test...")
        self.load_model()

        loaders = self.loaders
        if "test" not in loaders:
            raise ValueError("test loader is required for run_base_test")

        metrics_func = self._get_metrics_func()
        metric_name = getattr(metrics_func, "__name__", str(metrics_func))
        log.info(f"Base test metric function: {metric_name}")

        test_results = self._val_one_epoch(loader=loaders["test"], metrics_func=metrics_func)

        # yapf: disable
        # 指标字典更新
        test_results  = {"experiment_name" :self.run_cfg.experiment_name,
                         **test_results}
        # yapf: enable

        log.note(f"Test results: {test_results}")
        metrics_path = os.path.join(self.save_dir, "metrics.json")
        save_dict_as_json(test_results, metrics_path)
        log.info(f"Saved metrics JSON: {metrics_path}")
        return test_results

    def run_ipix_cfar_test(self):
        """Run IPIX-specific CFAR threshold test workflow."""
        log.info("Running IPIX CFAR test...")
        self.load_model()

        loaders = self.loaders
        if "test" not in loaders:
            raise ValueError("test loader is required for run_ipix_cfar_test")

        if "get_th" in loaders:
            th_loader = loaders["get_th"]
        elif "train" in loaders:
            log.warning("'get_th' loader is missing, fallback to train loader for threshold estimation")
            th_loader = loaders["train"]
        else:
            raise ValueError("Neither 'get_th' nor 'train' loader is available for threshold estimation")

        experiment_name = self.run_cfg.experiment_name
        metrics_func = self._get_metrics_func()
        metric_name = getattr(metrics_func, "__name__", str(metrics_func))
        log.info(f"IPIX CFAR metric function: {metric_name}")

        self.cfar_th = self.get_cfar_th(pfa=self.eval_cfg.pfa, loader=th_loader, f_name=experiment_name)

        self.net.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for signals, labels, *rest in loaders["test"]:

                inputs = signals.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(inputs)
                loss = self.loss_func(outputs, labels)
                total_loss += loss.item()

                preds = (outputs >= self.cfar_th).long()
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())

        test_results = {"loss": total_loss / max(len(loaders["test"]), 1)}

        if all_preds:
            preds = torch.cat(all_preds)
            labels = torch.cat(all_labels)
            test_results.update(metrics_func(preds, labels))

        # yapf: disable
        # 指标字典更新
        test_results  = {"experiment_name" :self.run_cfg.experiment_name,
                         "pfa" :self.eval_cfg.pfa,
                         "th":self.cfar_th,
                         **test_results}
        # yapf: enable

        tn = test_results.get("tn", 0)
        fp = test_results.get("fp", 0)
        fn = test_results.get("fn", 0)
        tp = test_results.get("tp", 0)

        print_confusion_matrix(experiment_name, tn, fp, fn, tp)
        log.note(f"[{experiment_name}] -> Pfa: {test_results.get('pfa', 0.0):.6f} | Pd: {test_results.get('pd', 0.0) * 100:.2f}%")
        metrics_path = os.path.join(self.save_dir, f"metrics_pfa{self.eval_cfg.pfa}.json")
        save_dict_as_json(test_results, metrics_path)
        log.info(f"Saved metrics JSON: {metrics_path}")
        return test_results

    def run_test(self):
        """Dispatch test workflow by ``eval.extra.test_mode``."""
        test_mode = self.eval_cfg.extra.get("test_mode", "base")
        log.info(f"Running test with mode: {test_mode}")

        match test_mode:
            case "base" | "run_base_test":
                return self.run_base_test()
            case "ipix_cfar" | "run_ipix_cfar_test":
                return self.run_ipix_cfar_test()
            case _:
                raise ValueError(f"Unsupported test_mode: {test_mode}")
