from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.models import resnet18, ResNet18_Weights

from create_dataset import office_Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT_ROOT, ".torch"))


class OfficeMultiTaskNet(nn.Module):
    """ResNet-18 trunk, adaptive pool, 512-d hidden block, and one linear head per task."""

    def __init__(self, tasks: List[str], class_num: int, pretrained: bool = True):
        super().__init__()
        if pretrained:
            try:
                backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception as exc:
                print(f"[WARN] Failed to load pretrained ResNet-18 ({exc}). Falling back to random initialization.")
                backbone = resnet18(weights=None)
        else:
            backbone = resnet18(weights=None)
        self.resnet_network = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.hidden = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.heads = nn.ModuleDict({task: nn.Linear(512, class_num) for task in tasks})
        nn.init.normal_(self.hidden[0].weight, mean=0.0, std=0.005)
        nn.init.constant_(self.hidden[0].bias, 0.1)

    def forward_task(self, x: torch.Tensor, task: str) -> torch.Tensor:
        z = self.resnet_network(x)
        z = torch.flatten(self.avgpool(z), 1)
        z = self.hidden(z)
        return self.heads[task](z)


def accuracy_from_cache(
    model: OfficeMultiTaskNet,
    caches: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    tasks: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    """Per-task accuracy on cached images, in domain order."""
    model.eval()
    task_acc: Dict[str, float] = {}
    with torch.no_grad():
        for task in tasks:
            images, labels = caches[task]
            correct = 0
            total = int(labels.shape[0])
            for start in range(0, total, batch_size):
                batch_x = images[start : start + batch_size].to(device, non_blocking=True)
                batch_y = labels[start : start + batch_size].to(device, non_blocking=True)
                pred = model.forward_task(batch_x, task).argmax(dim=1)
                correct += int((pred == batch_y).sum().item())
            task_acc[task] = 100.0 * correct / max(total, 1)
    return task_acc


def eval_split(
    model: OfficeMultiTaskNet,
    loaders: Dict[str, DataLoader],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    task_acc = {}
    with torch.no_grad():
        for task, loader in loaders.items():
            correct = 0
            total = 0
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                pred = model.forward_task(images, task).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.numel()
            task_acc[task] = 100.0 * correct / max(total, 1)
    return task_acc


class PooledPoints(Dataset):
    """One sequence S formed by concatenating per-domain training sets."""

    def __init__(self, parts: Sequence[Dataset]):
        self.parts = list(parts)
        self.offsets: List[int] = []
        total = 0
        for part in self.parts:
            self.offsets.append(total)
            total += len(part)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int):
        for offset, part in zip(self.offsets, self.parts):
            if index < offset + len(part):
                return part[index - offset]
        raise IndexError(index)


def regularity_rho(grad_matrix: torch.Tensor) -> float:
    """mu_min(U^T Q^T Q U), with columns of grad_matrix the objective gradients.
    """
    if grad_matrix.ndim != 2 or grad_matrix.shape[1] < 2:
        raise ValueError(f"Expected Q with M >= 2 columns, got shape {tuple(grad_matrix.shape)}")
    gram = grad_matrix.detach().to(dtype=torch.float64)
    gram = gram.transpose(0, 1).matmul(gram)
    m = gram.shape[0]
    ones = torch.ones(1, m, device=gram.device, dtype=torch.float64)
    _u, _s, vh = torch.linalg.svd(ones, full_matrices=True)
    basis = vh[1:].transpose(0, 1)
    restricted = basis.transpose(0, 1).matmul(gram).matmul(basis)
    smallest = torch.linalg.eigvalsh(restricted).min()
    # U^T Q^T Q U is positive semidefinite; clamp only negative roundoff.
    return float(smallest.clamp(min=0.0))


def _project_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto {lambda >= 0, sum lambda = 1}."""
    m = v.numel()
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    ind = torch.arange(1, m + 1, device=v.device, dtype=v.dtype)
    rho = int(torch.nonzero(u - cssv / ind > 0, as_tuple=False)[-1].item())
    theta = cssv[rho] / (rho + 1.0)
    return torch.clamp(v - theta, min=0.0)


def _mgda_lambda_two(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Exact minimizer of ||lambda_1 q1 + lambda_2 q2||^2 on the simplex."""
    g1 = q1.detach().to(dtype=torch.float64)
    g2 = q2.detach().to(dtype=torch.float64)
    g11 = torch.dot(g1, g1)
    g22 = torch.dot(g2, g2)
    g12 = torch.dot(g1, g2)
    denom = g11 + g22 - 2.0 * g12
    if float(denom) <= 0.0:
        lam1 = 0.5
    else:
        lam1 = float(((g22 - g12) / denom).clamp(0.0, 1.0))
    return torch.tensor([lam1, 1.0 - lam1], device=q1.device, dtype=q1.dtype)


def _min_norm_element_from2(v1v1, v1v2, v2v2):
    if v1v2 >= v1v1:
        return 0.999, v1v1
    if v1v2 >= v2v2:
        return 0.001, v2v2
    gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
    cost = v2v2 + gamma * (v1v2 - v2v2)
    return gamma, cost


def mgda_lambda(grad_matrix: torch.Tensor) -> torch.Tensor:
    """argmin_{lambda in Delta_M} ||Q lambda||^2.

    M = 2 uses the closed form. M > 2 uses the min-norm Frank-Wolfe solver.
    """
    m = grad_matrix.shape[1]
    if m == 2:
        return _mgda_lambda_two(grad_matrix[:, 0], grad_matrix[:, 1])

    grads = grad_matrix.detach().to(dtype=torch.float64).transpose(0, 1).contiguous()
    gram = grads.matmul(grads.transpose(0, 1))
    best_cost = float("inf")
    best_pair = None
    for i in range(m):
        for j in range(i + 1, m):
            gamma, cost = _min_norm_element_from2(gram[i, i], gram[i, j], gram[j, j])
            if cost < best_cost:
                best_cost = float(cost)
                best_pair = (i, j, gamma)
    sol = torch.zeros(m, dtype=torch.float64, device=gram.device)
    i, j, gamma = best_pair
    sol[i] = gamma
    sol[j] = 1.0 - gamma

    for _ in range(250):
        grad_dir = -gram.matmul(sol)
        proj_grad = grad_dir - grad_dir.sum() / m
        neg = proj_grad < 0
        pos = proj_grad > 0
        step = torch.ones((), dtype=torch.float64, device=gram.device)
        tm1 = -sol[neg] / proj_grad[neg]
        tm2 = (1.0 - sol[pos]) / proj_grad[pos]
        tm1 = tm1[tm1 > 1e-7]
        tm2 = tm2[tm2 > 1e-7]
        if tm1.numel() > 0:
            step = torch.minimum(step, tm1.min())
        if tm2.numel() > 0:
            step = torch.minimum(step, tm2.min())
        new_point = _project_simplex(sol + step * proj_grad)
        v1v1 = torch.dot(sol, gram.matmul(sol))
        v1v2 = torch.dot(sol, gram.matmul(new_point))
        v2v2 = torch.dot(new_point, gram.matmul(new_point))
        gamma, _cost = _min_norm_element_from2(v1v1, v1v2, v2v2)
        updated = gamma * sol + (1.0 - gamma) * new_point
        if torch.sum(torch.abs(updated - sol)) < 1e-8:
            sol = updated
            break
        sol = updated
    return _project_simplex(sol).to(device=grad_matrix.device, dtype=grad_matrix.dtype)


def choose_lambda(grad_matrix: torch.Tensor, mu_t: float) -> Tuple[torch.Tensor, str, float]:
    """Regular branch solves MGDA. Fallback uses the uniform weight and does not."""
    rho = regularity_rho(grad_matrix)
    m = grad_matrix.shape[1]
    if rho >= mu_t:
        return mgda_lambda(grad_matrix), "mgda", rho
    uniform = torch.full((m,), 1.0 / m, device=grad_matrix.device, dtype=grad_matrix.dtype)
    return uniform, "fallback", rho


def flatten_grad(loss: torch.Tensor, params: Sequence[torch.nn.Parameter], retain_graph: bool) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    pieces = []
    for param, grad in zip(params, grads):
        if grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(grad.reshape(-1))
    return torch.cat(pieces)


def assign_gradient_(params: Sequence[torch.nn.Parameter], direction: torch.Tensor) -> None:
    offset = 0
    for param in params:
        width = param.numel()
        param.grad = direction[offset : offset + width].view_as(param).clone()
        offset += width


def batch_size_at(t: int, b0: int, growth: float) -> int:
    """|Z_t| = b0 + floor(c t), which is Theta(t + 1) for c > 0 and b0 > 0."""
    return max(1, int(b0 + math.floor(growth * t)))


def mu_at(t: int, mu_scale: float) -> float:
    """mu_t = Theta((t + 1)^{-1/2})."""
    return float(mu_scale) / math.sqrt(t + 1)


def sample_batch(
    dataset: Dataset,
    k: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """k indices drawn i.i.d. uniformly with replacement."""
    sampler = RandomSampler(dataset, replacement=True, num_samples=k, generator=generator)
    loader = DataLoader(dataset, batch_size=k, sampler=sampler, num_workers=0)
    images, labels = next(iter(loader))
    return images, labels


def sample_cached(
    images: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """k i.i.d. uniform draws with replacement from the cached sequence S."""
    index = torch.randint(0, images.shape[0], (k,), generator=generator)
    return images.index_select(0, index), labels.index_select(0, index)


def load_or_build_cache(dataset: Dataset, tasks: Sequence[str], cache_path: str, log) -> Tuple[torch.Tensor, torch.Tensor]:
    if os.path.isfile(cache_path):
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        if blob.get("tasks") == list(tasks) and blob["images"].shape[0] == len(dataset):
            log(f"[INFO] Loaded cached training tensors from {cache_path}")
            return blob["images"], blob["labels"]
    log(f"[INFO] Caching {len(dataset)} training images into RAM")
    xs = []
    ys = []
    started = time.time()
    for i in range(len(dataset)):
        image, label = dataset[i]
        xs.append(image)
        ys.append(int(label))
        if (i + 1) % 1000 == 0 or i + 1 == len(dataset):
            log(f"[INFO] cached {i + 1}/{len(dataset)} images in {time.time() - started:.1f}s")
    images = torch.stack(xs)
    labels = torch.tensor(ys, dtype=torch.long)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"images": images, "labels": labels, "tasks": list(tasks)}, cache_path)
    log(f"[INFO] Wrote cache {cache_path} ({images.shape[0]} x {tuple(images.shape[1:])})")
    return images, labels


def split_sizes(n: int, n_parts: int) -> List[int]:
    """Split n examples across devices, keeping each shard large enough for BatchNorm."""
    parts = min(n_parts, max(1, n // 2))
    sizes = [n // parts] * parts
    for i in range(n % parts):
        sizes[i] += 1
    if any(size < 2 for size in sizes):
        return [n]
    return sizes


def broadcast_state_(primary: torch.nn.Module, replicas: Sequence[torch.nn.Module]) -> None:
    with torch.no_grad():
        for replica in replicas:
            for src, dst in zip(primary.parameters(), replica.parameters()):
                dst.data.copy_(src.data, non_blocking=True)
            for src, dst in zip(primary.buffers(), replica.buffers()):
                dst.data.copy_(src.data, non_blocking=True)


def average_buffers_(replicas: Sequence[torch.nn.Module]) -> None:
    if len(replicas) < 2:
        return
    with torch.no_grad():
        for bufs in zip(*(replica.buffers() for replica in replicas)):
            total = None
            for buf in bufs:
                value = buf.detach().to(device=bufs[0].device, dtype=torch.float32)
                total = value if total is None else total + value
            mean = (total / len(bufs)).to(dtype=bufs[0].dtype)
            for buf in bufs:
                buf.copy_(mean, non_blocking=True)


def shard_grads(replica: torch.nn.Module, task_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]], tasks: Sequence[str]):
    """One forward_task per domain batch."""
    device = next(replica.parameters()).device
    torch.cuda.set_device(device)
    params = [p for p in replica.parameters() if p.requires_grad]
    flats = []
    losses = []
    for task, (images, labels) in zip(tasks, task_batches):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        loss = F.cross_entropy(replica.forward_task(images, task), labels)
        losses.append(float(loss.detach()))
        flats.append(flatten_grad(loss, params, retain_graph=False))
    return flats, losses


def more_step(replicas: Sequence[torch.nn.Module], optimizer, batches: Sequence[Tuple[torch.Tensor, torch.Tensor]], tasks: Sequence[str], mu_t: float):
    batch_n = batches[0][0].shape[0]
    sizes = split_sizes(batch_n, len(replicas))
    active = list(replicas[: len(sizes)])
    if len(active) > 1:
        broadcast_state_(active[0], active[1:])
    per_replica = [[] for _ in active]
    for images, labels in batches:
        start = 0
        for replica_index, size in enumerate(sizes):
            per_replica[replica_index].append((images[start : start + size], labels[start : start + size]))
            start += size
    if len(active) == 1:
        results = [shard_grads(active[0], per_replica[0], tasks)]
    else:
        with ThreadPoolExecutor(max_workers=len(active)) as pool:
            futures = [
                pool.submit(shard_grads, replica, task_batches, tasks)
                for replica, task_batches in zip(active, per_replica)
            ]
            results = [future.result() for future in futures]
    average_buffers_(active)

    device = next(active[0].parameters()).device
    total_n = float(sum(sizes))
    grad_matrix = None
    task_losses = [0.0] * len(tasks)
    for (flats, losses), size in zip(results, sizes):
        weight = size / total_n
        stacked = torch.stack([grad.to(device=device, dtype=torch.float32) for grad in flats], dim=1)
        grad_matrix = stacked * weight if grad_matrix is None else grad_matrix + stacked * weight
        for index, value in enumerate(losses):
            task_losses[index] += weight * value

    params = [p for p in active[0].parameters() if p.requires_grad]
    lam, branch, rho = choose_lambda(grad_matrix, mu_t)
    direction = (grad_matrix * lam.unsqueeze(0)).sum(dim=1)
    optimizer.zero_grad(set_to_none=True)
    assign_gradient_(params, direction.detach())
    optimizer.step()
    scalar = sum(float(lam[i]) * task_losses[i] for i in range(len(tasks)))
    return scalar, branch, rho, lam


def run_more(args: argparse.Namespace) -> None:
    if not args.dataset_path:
        raise ValueError("--dataset-path is required to train.")
    if len(args.tasks) < 2:
        raise ValueError("MoRe needs at least two objectives.")
    if args.epochs < 0:
        raise ValueError("--epochs must be >= 0. Use 0 to train for --steps iterations.")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        n_visible = torch.cuda.device_count()
        n_gpus = n_visible if args.num_gpus <= 0 else min(args.num_gpus, n_visible)
        devices = [torch.device(f"cuda:{i}") for i in range(n_gpus)]
    else:
        devices = [torch.device("cpu")]
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train.log")
    log_file = open(log_path, "a", buffering=1)

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"[INFO] devices={devices} | M={len(args.tasks)} | tasks={args.tasks}")
    primary = OfficeMultiTaskNet(tasks=list(args.tasks), class_num=65, pretrained=args.pretrained).to(devices[0])
    replicas = [primary]
    for device in devices[1:]:
        replicas.append(copy.deepcopy(primary).to(device))
    for replica in replicas:
        replica.train()
    # Equation (17): constant step on Q lambda. No momentum and no weight decay.
    optimizer = SGD(primary.parameters(), lr=args.alpha, momentum=0.0, weight_decay=0.0)

    with open(os.path.join(args.save_dir, "args.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)

    def save_checkpoint(epoch: int, step: int) -> None:
        payload = {
            "model_state_dict": primary.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "args": vars(args),
        }
        if epoch > 0:
            path = os.path.join(args.save_dir, f"model_epoch_{epoch:04d}.pt")
            torch.save(payload, path)
            log(f"[INFO] Saved {path}")
        last = os.path.join(args.save_dir, "last_model.pt")
        torch.save(payload, last)

    def log_step(t: int, k: int, mu_t: float, rho: float, branch: str, scalar: float, lam: torch.Tensor) -> None:
        if t % args.log_every != 0:
            return
        lam_txt = ", ".join(f"{float(v):.3f}" for v in lam)
        log(
            f"t={t:06d} | M={len(args.tasks)} | Z={k} | mu={mu_t:.6e} | rho={rho:.6e} | "
            f"branch={branch} | loss={scalar:.4f} | lambda=[{lam_txt}]"
        )

    wall_start = time.time()
    if args.epochs > 0:
        cache_root = os.path.join(args.save_dir, "domain_cache")
        train_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        val_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for task in args.tasks:
            train_set = office_Dataset("office-home", args.dataset_path, task, "train", balanced=args.balanced)
            val_set = office_Dataset("office-home", args.dataset_path, task, "val", balanced=args.balanced)
            train_cache[task] = load_or_build_cache(
                train_set, [task], os.path.join(cache_root, f"{task}_train.pt"), log
            )
            val_cache[task] = load_or_build_cache(
                val_set, [task], os.path.join(cache_root, f"{task}_val.pt"), log
            )
        steps_per_epoch = min(train_cache[task][0].shape[0] // args.bs for task in args.tasks)
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        cap_txt = "none" if args.max_batch_size <= 0 else str(args.max_batch_size)
        log(
            f"[INFO] per-domain |Z_t|={args.bs}+floor({args.batch_growth}*t) cap={cap_txt} | "
            f"steps_per_epoch={steps_per_epoch} | alpha={args.alpha} (plain SGD) | "
            f"mu_t={args.mu_scale}/sqrt(t+1) | epochs={args.epochs}"
        )

        def batch_at(t: int) -> int:
            k = batch_size_at(t, args.bs, args.batch_growth)
            if args.max_batch_size > 0:
                k = min(k, args.max_batch_size)
            return k

        def domain_batches(k: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            return [sample_cached(images, labels, k, generator) for images, labels in (train_cache[task] for task in args.tasks)]

        t = 0
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            epoch_start = time.time()
            last_k = args.bs
            for _ in range(steps_per_epoch):
                mu_t = mu_at(t, args.mu_scale)
                last_k = batch_at(t)
                batches = domain_batches(last_k)
                scalar, branch, rho, lam = more_step(replicas, optimizer, batches, args.tasks, mu_t)
                log_step(t, last_k, mu_t, rho, branch, scalar, lam)
                epoch_loss += scalar
                t += 1
            train_acc = accuracy_from_cache(primary, train_cache, args.tasks, devices[0], args.bs)
            val_acc = accuracy_from_cache(primary, val_cache, args.tasks, devices[0], args.bs)
            for replica in replicas:
                replica.train()
            elapsed = time.time() - epoch_start
            train_avg = sum(train_acc.values()) / len(train_acc)
            val_avg = sum(val_acc.values()) / len(val_acc)
            train_txt = " ".join(f"{task}={train_acc[task]:.2f}" for task in args.tasks)
            val_txt = " ".join(f"{task}={val_acc[task]:.2f}" for task in args.tasks)
            log(
                f"[EPOCH {epoch:03d}] steps={steps_per_epoch} Z_end={last_k} "
                f"loss={epoch_loss / max(steps_per_epoch, 1):.4f} "
                f"train_acc={train_avg:.2f}% ({train_txt}) "
                f"val_acc={val_avg:.2f}% ({val_txt}) "
            )
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_checkpoint(epoch, t)
    else:
        parts = [
            office_Dataset("office-home", args.dataset_path, task, "train", balanced=args.balanced)
            for task in args.tasks
        ]
        dataset = PooledPoints(parts)
        if len(dataset) == 0:
            raise RuntimeError(f"No training images under {args.dataset_path} for tasks {args.tasks}.")
        cache_path = args.cache_path or os.path.join(args.save_dir, "pooled_cache.pt")
        cached_images, cached_labels = load_or_build_cache(dataset, args.tasks, cache_path, log)
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        cap_txt = "none" if args.max_batch_size <= 0 else str(args.max_batch_size)
        log(
            f"[INFO] shared Z_t | |Z_t|={args.bs}+floor({args.batch_growth}*t) cap={cap_txt} | "
            f"mu_t={args.mu_scale}/sqrt(t+1) | |S|={len(dataset)} | steps={args.steps}"
        )

        def batch_at(t: int) -> int:
            k = batch_size_at(t, args.bs, args.batch_growth)
            if args.max_batch_size > 0:
                k = min(k, args.max_batch_size)
            return k

        for t in range(args.steps):
            mu_t = mu_at(t, args.mu_scale)
            k = batch_at(t)
            images, labels = sample_cached(cached_images, cached_labels, k, generator)
            batches = [(images, labels) for _task in args.tasks]
            scalar, branch, rho, lam = more_step(replicas, optimizer, batches, args.tasks, mu_t)
            log_step(t, k, mu_t, rho, branch, scalar, lam)
        save_checkpoint(0, args.steps)
    log(f"[INFO] Training finished in {time.time() - wall_start:.1f}s. Log: {log_path}")
    log_file.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("MoRe (Algorithm 1) for M >= 2 objectives")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["Art", "Clipart", "Product", "Real_World"],
        help="One or more Office-Home domains. At least two are required.",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Iteration horizon used when --epochs is 0.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="If > 0, train this many epochs of min_domain_size//b0 steps, with |Z_t|=b0+floor(c t). Overrides --steps.",
    )
    parser.add_argument("--save-every", type=int, default=10, help="Save a checkpoint every this many epochs.")
    parser.add_argument("--num-gpus", type=int, default=0, help="GPUs to use. 0 means every visible GPU.")
    parser.add_argument("--max-batch-size", type=int, default=0, help="Optional cap on |Z_t|. 0 means no cap.")
    parser.add_argument(
        "--cache-path",
        type=str,
        default="",
        help="Path for the cached training tensors. Defaults to save-dir/pooled_cache.pt.",
    )
    parser.add_argument("--bs", type=int, default=32, help="b0 in |Z_t| = b0 + floor(c t).")
    parser.add_argument(
        "--batch-growth",
        type=float,
        default=1.0,
        help="c in |Z_t| = b0 + floor(c t). c > 0 gives the Theta(t+1) schedule.",
    )
    parser.add_argument("--alpha", type=float, default=1e-2, help="Constant step size in x <- x - alpha Q lambda.")
    parser.add_argument(
        "--mu-scale",
        type=float,
        default=800.0,
        help="Constant in mu_t = mu_scale / sqrt(t+1). 1000 matches the squared full-network gradient scale, where rho starts near 600 and falls to about 10.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/scratch/chuang80/office_outputs/more_algorithm",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Check rho, the branch, and the plain gradient step, then exit.",
    )
    return parser


def _self_check() -> None:
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.1, 2.7])
    pair = torch.stack((a, b), dim=1)
    rho = regularity_rho(pair)
    expected = 0.5 * float(torch.sum((a - b) ** 2))
    if abs(rho - expected) > 1e-8:
        raise AssertionError(f"rho {rho} != (1/2)||q1-q2||^2 {expected}")

    identical = torch.stack((torch.tensor([3.0, 0.0, 4.0]),) * 4, dim=1)
    if regularity_rho(identical) != 0.0:
        raise AssertionError("identical gradients must have rho 0, with no eigenvalue cutoff")

    eye = torch.eye(4)
    eye_rho = regularity_rho(eye)
    if abs(eye_rho - 1.0) > 1e-6:
        raise AssertionError(f"rho(I) should be 1, got {eye_rho}")

    g = torch.tensor([1.0, 0.0, 0.0])
    h = torch.tensor([0.0, 1.0, 0.0])
    paired = torch.stack((g, g, h, h), dim=1)
    paired_rho = regularity_rho(paired)
    if paired_rho > 1e-8:
        raise AssertionError(f"partially repeated gradients should be degenerate, rho={paired_rho}")

    # 2*rho would pass a threshold in (rho, 2*rho]; the paper's rho must fail it.
    _lam, branch, _rho = choose_lambda(pair, mu_t=0.08)
    if branch != "fallback":
        raise AssertionError("rho < mu_t must take the uniform fallback and skip MGDA")
    _lam, branch, _rho = choose_lambda(pair, mu_t=0.01)
    if branch != "mgda":
        raise AssertionError("rho >= mu_t must solve MGDA")
    _lam, branch, _rho = choose_lambda(eye, mu_t=0.5)
    if branch != "mgda" or _lam.numel() != 4:
        raise AssertionError("four objectives above the threshold must solve MGDA")

    ortho = torch.eye(3)
    lam = mgda_lambda(ortho)
    if not torch.allclose(lam, torch.full((3,), 1.0 / 3.0), atol=1e-4):
        raise AssertionError(f"orthogonal min-norm weight was {lam.tolist()}")
    if abs(float(lam.sum()) - 1.0) > 1e-5 or float(lam.min()) < -1e-6:
        raise AssertionError(f"lambda left the simplex: {lam.tolist()}")

    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.0, 0.0]]))
    opt = SGD(layer.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)
    target = torch.tensor([[1.0]])
    pred = layer(torch.tensor([[1.0, 0.0]]))
    loss = F.mse_loss(pred, target)
    grad = flatten_grad(loss, list(layer.parameters()), retain_graph=False)
    # d/dw0 of (w0 - 1)^2 is 2(w0-1) = -2 at w0=0; mean reduction is the same for one row.
    opt.zero_grad(set_to_none=True)
    assign_gradient_(list(layer.parameters()), grad)
    before = layer.weight.detach().clone()
    opt.step()
    expected_w = before - 0.1 * grad.view_as(before)
    if not torch.allclose(layer.weight, expected_w):
        raise AssertionError("SGD step is not x <- x - alpha g")
    print("[INFO] self-check passed")


if __name__ == "__main__":
    cli = build_parser().parse_args()
    if cli.self_check:
        _self_check()
    else:
        run_more(cli)
