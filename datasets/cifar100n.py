import os
import numpy as np
import torch
from torchvision.datasets import CIFAR100

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


def _get_classnames(root):
    try:
        tv_train = CIFAR100(root, train=True, download=False)
        if hasattr(tv_train, "classes") and len(tv_train.classes) == 100:
            return list(tv_train.classes)
    except Exception:
        pass
    return [str(i) for i in range(100)]


def _scan_indexed_images(base_dir):
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Missing directory: {base_dir}")

    index_to_path = {}
    index_to_label = {}
    for cls_name in sorted(os.listdir(base_dir)):
        cls_dir = os.path.join(base_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        try:
            label = int(cls_name)
        except Exception:
            continue

        for fname in os.listdir(cls_dir):
            lower = fname.lower()
            if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")):
                continue
            stem, _ = os.path.splitext(fname)
            if not stem.isdigit():
                continue
            idx = int(stem) - 1
            impath = os.path.join(cls_dir, fname)
            index_to_path[idx] = impath
            index_to_label[idx] = label

    if not index_to_path:
        raise RuntimeError(f"No indexed images found under: {base_dir}")

    max_idx = max(index_to_path.keys())
    num = max_idx + 1
    paths = [None] * num
    labels = np.empty((num,), dtype=np.int64)

    missing = []
    for i in range(num):
        p = index_to_path.get(i)
        if p is None:
            missing.append(i)
            continue
        paths[i] = p
        labels[i] = int(index_to_label[i])

    if missing:
        show = ", ".join(str(x) for x in missing[:20])
        more = "" if len(missing) <= 20 else f" ... (+{len(missing) - 20})"
        raise RuntimeError(
            "Missing indexed images under {} at indices: {}{}".format(base_dir, show, more)
        )

    return paths, labels


def _load_noisy_labels(labels_dir):
    noisy_pt = os.path.join(labels_dir, "CIFAR-100_human.pt")
    ordered_np = os.path.join(labels_dir, "CIFAR-100_human_ordered.npy")
    order_map_np = os.path.join(labels_dir, "image_order_c100.npy")

    clean_label = None
    noisy_label = None

    if os.path.exists(noisy_pt):
        try:
            try:
                noise_file = torch.load(noisy_pt, map_location="cpu", weights_only=False)
            except TypeError:
                noise_file = torch.load(noisy_pt, map_location="cpu")
            clean_label = noise_file.get("clean_label", noise_file.get("clean100", None))
            noisy_label = noise_file.get("noisy_label", noise_file.get("noisy100", None))
        except Exception:
            clean_label = None
            noisy_label = None

    if clean_label is None or noisy_label is None:
        if not (os.path.exists(ordered_np) and os.path.exists(order_map_np)):
            missing = []
            if not os.path.exists(noisy_pt):
                missing.append(noisy_pt)
            if not os.path.exists(ordered_np):
                missing.append(ordered_np)
            if not os.path.exists(order_map_np):
                missing.append(order_map_np)
            raise RuntimeError("Missing CIFAR-100N label files: " + ", ".join(missing))

        ordered = np.load(ordered_np, allow_pickle=True).item()
        order_map = np.load(order_map_np)
        inv_map = np.zeros_like(order_map)
        inv_map[order_map] = np.arange(order_map.shape[0])
        clean_label = np.array(ordered.get("clean_label"))[inv_map]
        noisy_label = np.array(ordered.get("noise_label"))[inv_map]
        return clean_label.astype(np.int64), noisy_label.astype(np.int64)

    return np.asarray(clean_label, dtype=np.int64), np.asarray(noisy_label, dtype=np.int64)


@DATASET_REGISTRY.register()
class CIFAR100N(DatasetBase):
    dataset_dir = "cifar-100n"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        if not os.path.isdir(self.dataset_dir):
            alt = os.path.join(root, "cifar100n")
            if os.path.isdir(alt):
                self.dataset_dir = alt

        self.image_train_dir = os.path.join(root, "cifar100", "train")
        self.image_test_dir = os.path.join(root, "cifar100", "test")

        train_paths, train_clean = _scan_indexed_images(self.image_train_dir)
        test_paths, test_clean = _scan_indexed_images(self.image_test_dir)
        classnames = _get_classnames(root)

        use_noisy = bool(getattr(cfg.DATASET, "NOISE_LABEL", True))
        noisy_label = None
        clean_label = train_clean
        labels_dir = os.path.join(self.dataset_dir, "data")

        if use_noisy:
            clean_label, noisy_label = _load_noisy_labels(labels_dir)
            if clean_label.shape[0] != train_clean.shape[0]:
                raise RuntimeError(
                    "Label length mismatch: {} vs {}".format(clean_label.shape[0], train_clean.shape[0])
                )

            match_ratio = float(np.mean(clean_label.astype(train_clean.dtype) == train_clean))
            if match_ratio < 0.9:
                ordered_np = os.path.join(labels_dir, "CIFAR-100_human_ordered.npy")
                order_map_np = os.path.join(labels_dir, "image_order_c100.npy")
                if not (os.path.exists(ordered_np) and os.path.exists(order_map_np)):
                    raise RuntimeError(
                        "Label/image order mismatch and missing reorder files: {}, {}".format(
                            ordered_np, order_map_np
                        )
                    )
                ordered = np.load(ordered_np, allow_pickle=True).item()
                order_map = np.load(order_map_np)
                inv_map = np.zeros_like(order_map)
                inv_map[order_map] = np.arange(order_map.shape[0])
                clean_label = np.array(ordered.get("clean_label"))[inv_map].astype(np.int64)
                noisy_label = np.array(ordered.get("noise_label"))[inv_map].astype(np.int64)

        train = []
        for i, impath in enumerate(train_paths):
            gt = int(train_clean[i])
            lab = int(noisy_label[i]) if use_noisy else gt
            cname = classnames[lab] if 0 <= lab < len(classnames) else str(lab)
            item = Datum(impath=impath, label=lab, classname=cname)
            item.gttarget = gt
            item.target = int(noisy_label[i]) if use_noisy else lab
            train.append(item)

        test = []
        for i, impath in enumerate(test_paths):
            gt = int(test_clean[i])
            cname = classnames[gt] if 0 <= gt < len(classnames) else str(gt)
            item = Datum(impath=impath, label=gt, classname=cname)
            test.append(item)

        # optional few-shot sampling
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, test=test)
