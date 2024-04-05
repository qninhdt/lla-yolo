# convert exdark to coco format
import os
import imagesize
from PIL import Image
import json

classnames = [
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
]

classname_to_id = {v: k + 1 for k, v in enumerate(classnames)}

data_root = "./data/exdark"


def extract():
    img_map = {}

    with open(data_root + "/imageclasslist.txt") as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith("Name"):
                continue

            line = line.strip()
            line = line.split(" ")

            img_name = line[0].lower()
            img_class = int(line[1])
            img_phase = int(line[4])

            img_map[img_name] = (img_class, img_phase)

        categories = [{"id": k + 1, "name": v} for k, v in enumerate(classnames)]

        results = [
            {"images": [], "annotations": [], "categories": categories}.copy()
            for _ in range(3)
        ]

        instance_count = 0
        for img_name, shit in img_map.items():
            img_class, img_phase = shit
            img_id = int(img_name.split("_")[1].split(".")[0])
            img_classname = classnames[img_class - 1]
            img_file = os.path.join(data_root, "images", img_classname, img_name)
            img_anno_file = os.path.join(
                data_root, "annotations", img_classname, img_name.split(".")[0] + ".txt"
            )

            instances = []
            with open(img_anno_file) as f:
                lines = f.readlines()

                for line in lines:
                    if line.startswith("%"):
                        continue

                    instance_count += 1

                    line = line.strip().split(" ")

                    category = classname_to_id[line[0]]
                    x1, y1, w, h = map(float, line[1:5])

                    instances.append(
                        {
                            "iscrowd": 0,
                            "bbox": [x1, y1, w, h],
                            "category_id": category,
                            "area": w * h,
                            "id": instance_count,
                            "image_id": img_id,
                        }
                    )

            width, height = imagesize.get(img_file)
            results[img_phase - 1]["images"].append(
                {
                    "id": img_id,
                    "file_name": img_name,
                    "width": width,
                    "height": height,
                }
            )

            results[img_phase - 1]["annotations"].extend(instances)

    json.dump(
        results[0],
        open(os.path.join("./data/exdark_coco/annotations", "train.json"), "w"),
        indent=4,
    )
    json.dump(
        results[1],
        open(os.path.join("./data/exdark_coco/annotations", "val.json"), "w"),
        indent=4,
    )
    json.dump(
        results[2],
        open(os.path.join("./data/exdark_coco/annotations", "test.json"), "w"),
        indent=4,
    )

    # copy images with os.walk
    # for root, dirs, files in os.walk(data_root + "/images"):
    #     for file in files:
    #         if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
    #             img = Image.open(os.path.join(root, file))
    #             img.save(os.path.join("./data/exdark_coco/images", file))


extract()
