import argparse
import json
import numpy as np

__author__ = 'Suriya Narayanan Lakshmanan'
__copyright__ = 'Copyright (c) 2018, Cyngn Inc.'
__email__ = ''
__license__ = ''


def parse_args():
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, help='path to the label dir')
    parser.add_argument('--det_path', type=str, help='path to output detection file')
    args = parser.parse_args()

    return args


def label2det(frames, json_name):
    coco_dataset = {}
    coco_dataset["info"] = "Conversion of BDD dataset into COCO format"
    coco_dataset["licenses"] = "None"
    coco_dataset["images"] = []
    coco_dataset["annotations"] = []
    coco_dataset["categories"] = [{"name": "bus", "id": 1}, \
                                  {"name": "traffic light", "id": 2}, \
                                  {"name": "traffic sign", "id": 3}, \
                                  {"name": "person", "id": 4}, \
                                  {"name": "bike", "id": 5}, \
                                  {"name": "truck", "id": 6}, \
                                  {"name": "motor", "id": 7}, \
                                  {"name": "car", "id": 8}, \
                                  {"name": "train", "id": 9}, \
                                  {"name": "rider", "id": 10} ]

    CATEGORY_MAP = {}
    for dt in coco_dataset["categories"]:
        CATEGORY_MAP[dt["name"]] = dt["id"]

    image_id = 1
    annotation_id = 1
    for frame in frames:
        coco_image = {}
        coco_image["file_name"] = frame['name']
        coco_image["id"] = image_id
        coco_image["calib"] = np.eye(4).tolist()

        coco_dataset["images"].append(coco_image)

        for label in frame['labels']:
            if 'box2d' not in label:
                continue
            xy = label['box2d']
            if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
                continue

            coco_annotation = {}
            coco_annotation["image_id"] = image_id
            coco_annotation["id"] = annotation_id
            coco_annotation["category_id"] = CATEGORY_MAP[label["category"]]
            coco_annotation["dim"] = None
            width = xy['x2']-xy['x1']
            height = xy['y2']-xy['y1']
            coco_annotation["bbox"] = [xy['x1'], xy['y1'], width, height]
            coco_annotation["depth"] = None
            coco_annotation["alpha"] = 0
            coco_annotation["truncated"] = 0
            coco_annotation["occluded"] = 0
            coco_annotation["location"] = None
            coco_annotation["rotation_y"] = 0
            coco_annotation["iscrowd"] = 0
            coco_annotation["area"] = width * height

            coco_dataset["annotations"].append(coco_annotation)

            annotation_id += 1

        image_id += 1
    with open(json_name, "w") as f:
        json.dump(coco_dataset, f)

def convert_labels(label_path, det_path):
    frames = json.load(open(label_path, 'r'))
    label2det(frames, det_path)

def main():
    args = parse_args()
    convert_labels(args.label_path, args.det_path)


if __name__ == '__main__':
    main()
