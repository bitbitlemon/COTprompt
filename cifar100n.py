import os
import urllib.request
import numpy as np
import torch
from torchvision.datasets import CIFAR100

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing, check_isfile

try:
    # helper to prepare cifar10/cifar100 images into folders
    from Dassl.pytorch.datasets.ssl.cifar10_cifar100_svhn import download_and_prepare
except Exception:
    download_and_prepare = None


@DATASET_REGISTRY.register()
class CIFAR100N(DatasetBase):
    dataset_dir = "cifar-100n"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # image folders prepared by helper script
        self.image_train_dir = os.path.join(root, "cifar100", "train")
        self.image_test_dir = os.path.join(root, "cifar100", "test")

        # ensure images exist; create from torchvision if missing
        need_prepare = (not os.path.isdir(self.image_train_dir)) or (not os.path.isdir(self.image_test_dir))
        if need_prepare and download_and_prepare is not None:
            mkdir_if_missing(root)
            print(f"Preparing CIFAR-100 images under {os.path.join(root,'cifar100')}")
            download_and_prepare("cifar100", root)

        # load torchvision datasets to access labels and class names
        tv_train = CIFAR100(root, train=True, download=True)
        tv_test = CIFAR100(root, train=False, download=True)
        classnames = tv_train.classes  # 100 fine-grained class names

        # ensure CIFAR-100N noisy label file is present
        labels_dir = os.path.join(self.dataset_dir, "data")
        mkdir_if_missing(labels_dir)
        noisy_pt = os.path.join(labels_dir, "CIFAR-100_human.pt")
        if not os.path.exists(noisy_pt):
            urls = [
                "https://mirrors.tuna.tsinghua.edu.cn/github-raw/UCSC-REAL/cifar-10-100n/master/data/CIFAR-100_human.pt",
                "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/master/data/CIFAR-100_human.pt",
            ]
            last_err = None
            for url in urls:
                print(f"Downloading CIFAR-100N labels from {url} to {noisy_pt}")
                try:
                    urllib.request.urlretrieve(url, noisy_pt)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    continue
            if last_err is not None:
                raise RuntimeError(f"Failed to download CIFAR-100N label file: {last_err}")

        clean_label = None
        noisy_label = None
        if os.path.exists(noisy_pt):
            try:
                noise_file = torch.load(noisy_pt, map_location="cpu", weights_only=False)
                clean_label = noise_file.get("clean_label", noise_file.get("clean100", None))
                noisy_label = noise_file.get("noisy_label", noise_file.get("noisy100", None))
            except Exception:
                clean_label = None
                noisy_label = None
        if clean_label is None or noisy_label is None:
            # Fallback: use TFDS-ordered numpy labels + image order mapping
            ordered_np = os.path.join(labels_dir, "CIFAR-100_human_ordered.npy")
            order_map_np = os.path.join(labels_dir, "image_order_c100.npy")
            if not os.path.exists(ordered_np) or not os.path.exists(order_map_np):
                urls_np = [
                    ("https://mirrors.tuna.tsinghua.edu.cn/github-raw/UCSC-REAL/cifar-10-100n/master/data/CIFAR-100_human_ordered.npy", ordered_np),
                    ("https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/master/data/CIFAR-100_human_ordered.npy", ordered_np),
                    ("https://mirrors.tuna.tsinghua.edu.cn/github-raw/UCSC-REAL/cifar-10-100n/master/data/image_order_c100.npy", order_map_np),
                    ("https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/master/data/image_order_c100.npy", order_map_np),
                ]
                for i in range(0, len(urls_np), 2):
                    url1, dst1 = urls_np[i]
                    url2, dst2 = urls_np[i+1]
                    try:
                        if not os.path.exists(dst1):
                            print(f"Downloading {url1} -> {dst1}")
                            urllib.request.urlretrieve(url1, dst1)
                        if not os.path.exists(dst2):
                            print(f"Downloading {url2} -> {dst2}")
                            urllib.request.urlretrieve(url2, dst2)
                        break
                    except Exception:
                        continue
            ordered = np.load(ordered_np, allow_pickle=True).item()
            order_map = np.load(order_map_np)
            inv_map = np.zeros_like(order_map)
            inv_map[order_map] = np.arange(order_map.shape[0])
            clean_label = np.array(ordered.get("clean_label"))
            noisy_label = np.array(ordered.get("noise_label"))
            clean_label = clean_label[inv_map]
            noisy_label = noisy_label[inv_map]

        # build train items following torchvision order
        train = []
        for i in range(len(tv_train)):
            true_lab = int(tv_train.targets[i])
            cname = classnames[int(noisy_label[i])]
            impath = os.path.join(self.image_train_dir, str(true_lab).zfill(3), str(i + 1).zfill(5) + ".jpg")
            if not check_isfile(impath):
                # if images were not extracted to files, save on-the-fly
                img, _ = tv_train[i]
                mkdir_if_missing(os.path.dirname(impath))
                img.save(impath)
            item = Datum(impath=impath, label=int(noisy_label[i]), classname=cname)
            item.gttarget = int(clean_label[i])
            item.target = int(noisy_label[i])
            train.append(item)

        # build test items with clean labels
        test = []
        for i in range(len(tv_test)):
            true_lab = int(tv_test.targets[i])
            cname = classnames[true_lab]
            impath = os.path.join(self.image_test_dir, str(true_lab).zfill(3), str(i + 1).zfill(5) + ".jpg")
            if not check_isfile(impath):
                img, _ = tv_test[i]
                mkdir_if_missing(os.path.dirname(impath))
                img.save(impath)
            item = Datum(impath=impath, label=true_lab, classname=cname)
            test.append(item)

        # optional few-shot sampling
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, test=test)
