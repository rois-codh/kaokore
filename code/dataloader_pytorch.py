#!/usr/bin/env python3

import csv
import os
from PIL import Image
from torch.utils.data import Dataset


def verify_str_arg(value, valid_values):
    assert value in valid_values
    return value


def image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def load_labels(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        return [{
            headers[column_index]: row[column_index]
            for column_index in range(len(row))
        }
                for row in reader]


class Kaokore(Dataset):

    def __init__(self, root, split='train', category='gender', transform=None):
        self.root = root = os.path.expanduser(root)

        self.split = verify_str_arg(split, ['train', 'dev', 'test'])

        self.category = verify_str_arg(category, ['gender', 'status'])

        labels = load_labels(os.path.join(root, 'labels.csv'))
        self.entries = [
            (label_entry['image'], int(label_entry[category]))
            for label_entry in labels
            if label_entry['set'] == split and os.path.exists(
                os.path.join(self.root, 'images_256', label_entry['image']))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        image_filename, label = self.entries[index]

        image_filepath = os.path.join(self.root, 'images_256', image_filename)
        image = image_loader(image_filepath)
        if self.transform is not None:
            image = self.transform(image)

        return image, label
