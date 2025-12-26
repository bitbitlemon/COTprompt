import os
import pickle
import urllib.request
import tarfile
from torchvision.datasets import Caltech101 as TVCaltech101

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):

    dataset_dir = "caltech-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if not os.path.exists(self.image_dir):
            mkdir_if_missing(self.dataset_dir)
            dst = os.path.join(self.dataset_dir, "101_ObjectCategories.tar.gz")
            if not os.path.exists(dst):
                url = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
                try:
                    urllib.request.urlretrieve(url, dst)
                except Exception:
                    try:
                        TVCaltech101(root=self.dataset_dir, download=True)
                    except Exception as e2:
                        raise RuntimeError(str(e2))
            if os.path.exists(dst):
                try:
                    with tarfile.open(dst, "r:gz") as tf:
                        tf.extractall(self.dataset_dir)
                except Exception:
                    pass

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
                    # normalize impath for cross-platform runs (Windows/Linux)
                    marker = "101_ObjectCategories"
                    def _normalize(items):
                        for item in items:
                            p = item.impath
                            rel = p
                            if marker in p:
                                idx = p.find(marker)
                                rel = p[idx + len(marker):]
                                if rel.startswith("\\") or rel.startswith("/"):
                                    rel = rel[1:]
                            rel = rel.replace("\\", "/")
                            item._impath = os.path.join(self.image_dir, rel)
                        return items
                    train = _normalize(train)
                    val = _normalize(val)
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
