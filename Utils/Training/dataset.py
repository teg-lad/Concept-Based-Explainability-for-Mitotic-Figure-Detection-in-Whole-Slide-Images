import os
from pathlib import Path
import json

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# Create a dataset class inheriting from the Dataset
class MidogDataset(Dataset):
    """
    Class for implementing a dataset for the data for training.
    """

    def __init__(self, root, transforms=None):
        """
        root - Path to the directory storing the data
        transforms - transform method to be applied to the data
        """

        # Store the root and transforms as class variables.
        self.root = root
        self.transforms = transforms

        # load all image files, sorting them to ensure that they are aligned.
        # Save the images into an instance variable.
        self.imgs = list(sorted(os.listdir(Path(root, "data"))))

        # Open the training json and load in the data.
        with open(os.path.join(root, "training.json")) as t:
            training_data = json.load(t)

        # Store the data into an instance variable.
        self.data = training_data["images"]

    def __getitem__(self, idx):
        """
        Returns the data for a given index

        idx - The index of the data item to be returned
        """

        # Get the image path and load the image into memory.
        img_path = Path(self.root, "data", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # Parse the image path to get the image and tild id.
        image_id, tile_id = self.imgs[idx].split(".")[0].split("_")

        # Get the tile info for the given image and tile id.
        tile_info = self.get_tile_info(image_id, tile_id)

        # Extract the bounding boxes from the tile annotations.
        boxes = []
        for anno in tile_info["annotations"]:
            left, bottom, right, top = anno["bounding_box"].values()

            boxes.append([left, bottom, right, top])

        # Get the number of objects so we can create the labels.
        num_objs = len(boxes)

        # There is only one class, mitotic figure, so all labels are 1.
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Convert the boxes to a tensor of int32.
        # Note: int 16 is enough to hold the information we have but the model expects the targets in int32 format.
        boxes = torch.as_tensor(boxes, dtype=torch.int32)

        # Create the target dictionary to be returned for a data item.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["tile_id"] = tile_id

        # If there are any transforms defined we can carry them out here.
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        # The number of data items we have the length of our list of image paths.
        # We could have used the individual annotations as a unit of data but this adds additional complexity and
        # model training is not the primary objective of this project.
        return len(self.imgs)

    def get_tile_info(self, image_id, tile_id):
        """
        Return the dictionary that contains information relating to the given image and tile id.

        image_id - The int id of the image we want the data for.
        tile_id - The int id of the tile we want the data for.
        """

        # Get the dictionary with information on the given image.
        image_info = next((image for image in self.data if image["image_id"] == int(image_id)), None)

        # Within this image dictionary, get the dictionary that relates to the given tile id.
        tile_info = next((tile for tile in image_info["tiles"] if tile["tile_id"] == int(tile_id)), None)

        return tile_info


# Define a transformation for the dataset so can have the image as a tensor for use with the model.
def transforms(img, target):
    return T.ToTensor()(img), target
