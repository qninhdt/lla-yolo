import mmengine

import os
from mmdet.datasets import BaseDetDataset
from ..registry import DATASETS

import imagesize


@DATASETS.register_module()
class ExDarkDataset(BaseDetDataset):

    METAINFO = {
        "classes": (
            "Bicycle",
            "Boat",
            "Bottle",
            "Bus",
            "Car",
            "Cat",
            "Chair",
            "Cup",
            "Dog",
            "Motorbike",
            "People",
            "Table",
        ),
        "palette": [
            (220, 20, 60),
            (119, 11, 32),
            (0, 0, 142),
            (0, 0, 230),
            (0, 0, 70),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 230),
            (0, 0, 110),
            (0, 0, 230),
            (0, 0, 230),
            (0, 0, 230),
        ],
    }

    def __init__(self, *args, batch_shapes_cfg=None, **kwargs):
        self.batch_shapes_cfg = batch_shapes_cfg
        super().__init__(*args, **kwargs)

    def load_data_list(self):
        phase_id = 0 if "train" == self.ann_file else 1 if "val" == self.ann_file else 2

        class_to_id = {v: k for k, v in enumerate(self.METAINFO["classes"])}

        img_map = {}

        with open(self.data_root + "/imageclasslist.txt") as f:
            lines = f.readlines()

            for line in lines:
                if line.startswith("Name"):
                    continue

                line = line.strip()
                line = line.split(" ")

                img_name = line[0]
                img_class = int(line[1]) - 1
                img_phase = int(line[2]) - 1

                if img_phase != phase_id:
                    continue

                img_map[img_name] = img_class

        data_infos = []
        for img_name, img_class in img_map.items():
            img_id = int(img_name.split("_")[1].split(".")[0])
            img_classname = self.METAINFO["classes"][img_class]
            img_file = os.path.join(self.data_root, "images", img_classname, img_name)
            img_anno_file = os.path.join(
                self.data_root, "annotations", img_classname, img_name + ".txt"
            )

            instances = []
            with open(img_anno_file) as f:
                lines = f.readlines()

                for line in lines:
                    if line.startswith("%"):
                        continue

                    line = line.strip().split(" ")

                    label = class_to_id[line[0]]
                    x1, y1, w, h = map(int, line[1:5])
                    x2, y2 = x1 + w, y1 + h

                    instances.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "bbox_label": label,
                        }
                    )

            width, height = imagesize.get(img_file)
            data_infos.append(
                {
                    "img_path": os.path.join(self.data_root, "images", img_name),
                    "img_id": img_id,
                    "width": width,
                    "height": height,
                    "instances": instances,
                }
            )

        return data_infos
