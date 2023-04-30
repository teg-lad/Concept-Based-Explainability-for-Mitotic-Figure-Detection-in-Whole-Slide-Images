""" collection of various helper functions for running ACE"""

import sys
import os
import copy
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path

import torchvision

# Allows us to import the ModelWrapper.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.model_wrapper import ModelWrapper


class MyModel:
    """
    This class holds the PyTorch model and functions to run the model and return gradients for various inputs.
    """

    def __init__(self, model, layers):
        """
        Initialises the object with a model and the layers needed for activations and gradients.
        :param model: string specifying the model to use, which is then loaded in the init.
        :param layers: The layers that hooks are to be inserted into.
        """

        # If the model is "mitotic", load our trained model.
        if model == "mitotic":

            # load a model pre-trained on COCO
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

            # replace the classifier with a new one, that has
            # num_classes which is user-defined
            num_classes = 2  # 1 class (mitotic figure) + background

            # get number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features

            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            # Load in the best model by specifying the path to the save.
            checkpoint = torch.load("D:/DS/DS4/Project/model_saves/2023_03_04_17_22_33_11.pth")

            # Take the model that we have already initialized and load the state_dict into it.
            model.load_state_dict(checkpoint["model_state_dict"])

            # Pass the model into a wrapper that manages adding hooks.
            self.model = ModelWrapper(model, layers)

        # If no type is specified, use the default mode.
        else:

            # Pass the model into a wrapper that manages adding hooka.
            self.model = ModelWrapper(torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT"),
                                      layers)

        # Set the model to evaluation.
        self.model.eval()

        # Move the model to the device, cuda if available, otherwise it goes on the cpu.
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def run_examples(self, imgs, get_mean=True):
        """
        This functions takes images and passes them through the model and returns the resultant activations.
        :param imgs: A numpy array of images to be passed to the model.
        :param get_mean: Specify which layers should have the returned activations averaged across channels. This is to
        save memory and to allow for PCA to be run on the activations.
        :return: The activations for the images in each of the bottleneck layers.
        """

        # If get mean is not a list
        if type(get_mean) != list:
            # Make it a list the length of the layers we want to pull activations from.
            get_mean = len(self.model.layers) * [get_mean]

        # Convert the numpy array images to a tensor.
        tensor_imgs = torch.from_numpy(imgs).float()

        # If the shape is only 3 dimensions, i.e. had only one image in the array, add an empty dimension for batch
        # size.
        if len(tensor_imgs.shape) < 4:
            tensor_imgs = tensor_imgs[None, :]

        # Rearrange the dimensions.
        tensor_imgs = tensor_imgs.permute(0, 3, 1, 2)

        # Move the images to the device.
        tensor_imgs = tensor_imgs.to(self.device)

        # Pass the images to the model and ensure no gradients are calculated or references stored in the graph.
        with torch.no_grad():

            # Pass the images.
            self.model(tensor_imgs)

            # Pull the intermediate activations out.
            acts = self.model.intermediate_activations

        # Flatten and average the activations.
        output = flatten_and_average(acts, get_mean)

        # Return the output.
        return output

    def label_to_id(self, label):
        """
        This function is used to return the id for the given class label.
        :param label: The class label as a string.
        :return:
        """

        # Dictionary containing the class labels with their values.
        default = {"tennis racket": 43,
                   "mitotic figure": 1}

        return default[label]

    def get_gradient(self, imgs, class_id, get_mean=True):
        """
        This function returns the gradients for the given images.
        :param imgs: A tensor or list of images.
        :param class_id: The class id that the gradients should be computed with respect to.
        :param get_mean: Boolean list that specifies which layers should be averaged when returned.
        :return: A dictionary of the gradients for the images in each layer.
        """

        # If get_mean is not a list, convert it to one of length layers which we want to get the gradients for.
        if type(get_mean) != list:
            get_mean = len(self.model.layers) * [get_mean]

        # Stack the image tensors to form a batch.
        tensor_imgs = torch.stack(imgs)

        # Delete the images to free up memory.
        del imgs

        # Move the tensor images to the device the model is on.
        tensor_imgs = tensor_imgs.to(self.device)

        # Pass the images to the model.
        self.model(tensor_imgs)

        # Delete the image tensors.
        del tensor_imgs

        # Generate the gradients for the image for the given class id.
        gradients, info = self.model.generate_gradients(class_id)

        # If there are gradients returned process them (If no predictions are made there will be no gradients.)
        if len(list(gradients.values())[0]) > 0:

            # Flatten and average the gradients that are returned.
            output = flatten_and_average(gradients, get_mean)

        # If there are no gradients then the output will be None.
        else:
            output = None

        # Return the output and the info return from generating the gradients.
        return output, info


def flatten_and_average(dict, get_mean):
    """
    This function takes a dictionary of activations or gradients and flattens the tensors. Additionally, this function
    can get the average across the channels in the activations or gradients to reduce the dimensionality.
    :param dict: Dictionary of activations or gradients.
    :param get_mean: List specifying if the channels should be averaged in each layer.
    :return: The flattened and averaged arrays.
    """

    # Create an empty dictionary to store the output.
    flattened = {}

    # For every layer we retrieved activations or gradients for.
    for k, v, channel_mean in zip(dict.keys(), dict.values(), get_mean):

        # If this layer has been specified as one to get the average of
        if channel_mean:
            # Average the tensor across dimension one and flatten from at the first dimension to preserve the batch and
            # have the activations or gradients for each image separate.
            flattened[k] = torch.flatten(torch.mean(v.detach(), dim=1).cpu(), start_dim=1)

        # If it has not be specified as one to average.
        else:

            # Only flatten the tensor.
            flattened[k] = torch.flatten(v.detach().cpu(), start_dim=1)

    # Return a dictionary with the layer as the key and a list of numpy arrays.
    return {k: list(v.numpy()) for k, v in flattened.items()}


def create_directories(output_path, remove_old=False):
    """
    This function creates the directories in which the output for ConceptDiscovery is stored.
    :param output_path: The path in which to create these directories.
    :param remove_old: Should old output that was previouslt generated be removed?
    """

    # Create an output directory for our data
    output = Path(output_path)

    # Create the relevant subdirectories.
    discovered_concepts_dir = output / 'concepts/'
    results_dir = output / 'results/'
    cavs_dir = output / 'cavs/'
    activations_dir = output / 'acts/'
    results_summaries_dir = output / 'results_summaries/'

    # If the directory exists, and we want it deleted, delete it.
    if output.exists() and remove_old:
        shutil.rmtree(output)

    # Make all the directories
    discovered_concepts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    cavs_dir.mkdir(parents=True, exist_ok=True)
    activations_dir.mkdir(parents=True, exist_ok=True)
    results_summaries_dir.mkdir(parents=True, exist_ok=True)


def return_param(param_dict, param_name, default):
    """
    This function is a helper to access parameter dictionaries.
    :param param_dict: Dictionary containing parameters.
    :param param_name: The name of the parameter to access in the dictionary.
    :param default: The default value to use if the parameter is not in the dictionary.
    :return: The selected parameter or the default.
    """

    # If the parameter name is in the dictionary.
    if param_name in param_dict.keys():

        # Return this value.
        return param_dict[param_name]

    # Otherwise, return the default.
    else:
        return default


def load_image_from_file(filename, shape):
    """
    This function loads an image from file and resizes it to the specified shape. If the file cannot be opened None is
    returned.
    :param filename: The location of the file to open.
    :param shape: The shape the image should be resized to.
    :return: The image or None if unsuccessful.
    """

    # If the filename cannot be found, return None.
    if not Path(filename).exists():
        print('Cannot find file: {}'.format(filename))
        return None

    # Try to open and scale the image.
    try:
        img = np.array(Image.open(filename).resize(shape, Image.BILINEAR))

    # Catch and print any exception.
    except Exception as e:
        print(e)
        return None

    # Return the image.
    return img


def save_concepts(cd, bs=32):
    """
    Saves the patches and superpixels from the discovered concepts.
    :param cd: The ConceptDiscovery instance.
    :param bs: The number of images to be saved at one time.
    :return:
    """

    # For every bottleneck layer.
    for bn in cd.bottlenecks:

        # For every concept that we have discovered.
        for concept in cd.dic[bn]['concepts']:

            # Get the directory in which these images will be saved.
            patches_dir = Path(cd.discovered_concepts_dir, bn, concept + '_patches')
            images_dir = Path(cd.discovered_concepts_dir, bn, concept)

            # Create the specified directories.
            Path(patches_dir).mkdir(parents=True, exist_ok=True)
            Path(images_dir).mkdir(parents=True, exist_ok=True)

            # For every batch in the concept images.
            for i in range(int(len(cd.dic[bn][concept]['patches']) / bs) + 1):

                # Open the patches in this batch.
                patches = np.array(
                    [np.array(Image.open(img)) for img in cd.dic[bn][concept]['patches'][i * bs:(i + 1) * bs]])

                # Open the superpixels for this concept.
                superpixels = np.array(
                    [np.array(Image.open(img)) for img in cd.dic[bn][concept]['images'][i * bs:(i + 1) * bs]])

                # Get the image numbers from the concept dictionary
                image_numbers = cd.dic[bn][concept]['image_numbers'][i * bs:(i + 1) * bs]

                # Use these image numbers to create the addresses to save the images to.
                image_addresses = [images_dir / f"{img_num}.png" for img_num in image_numbers]
                patch_addresses = [patches_dir / f"{img_num}.png" for img_num in image_numbers]

                # Save the images.
                save_images(patch_addresses, patches)
                save_images(image_addresses, superpixels)


def save_discovery_images(cd, bs=32, save_context=False):
    """
    Save the concept discovery images to the output.
    :param cd: The ConceptDiscovery instance.
    :param bs: The batch size for saving the images.
    :param save_context: Boolean specifying if the context discovery images are to be saved.
    """

    # If there are no discovery images or we specify we want to save the context images.
    if cd.discovery_images is None or save_context:

        # If we want to save the context images.
        if save_context:

            # Get the list of context discovery image paths.
            concept_dir = cd.source_dir / cd.target_class / "context_discovery"

        else:
            # Otherwise, get the discovery images.
            concept_dir = cd.source_dir / cd.target_class / "discovery"

        # Save the list of discovery images paths
        self.discovery_images = list(concept_dir.iterdir())

    # Create the output directory.
    image_dir = cd.discovered_concepts_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)

    # For every batch of images in the discovery images.
    for i in range(int(len(cd.discovery_images) / bs) + 1):

        # Get the current batch.
        current_batch = cd.discovery_images[i * bs:(i + 1) * bs]

        # Load the images.
        images = np.array([load_image_from_file(img, cd.resize_dims) for img in current_batch])

        # Convert the images to uint8.
        converted_images = (images * 256).astype(np.uint8)

        # Get the list of addresses to save the images under.
        image_addresses = [image_dir / f"{img.name}.png" for img in current_batch]

        # Save the images.
        save_images(image_addresses, converted_images)


def save_images(addresses, images):
    """
    Takes images and a set of addresses and saves the images under the corresponding addresses supplied.
    :param addresses: List of image paths to save images to.
    :param images: List of images to save.
    :return:
    """

    # If we don't have a list of addresses.
    if not isinstance(addresses, list):

        # Create a list and number them starting from 0.
        image_addresses = []
        for i, image in enumerate(images):
            image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.png'
            image_addresses.append(os.path.join(addresses, image_name))
        addresses = image_addresses

    # Assert that we have an address for each image.
    assert len(addresses) == len(images), 'Invalid number of addresses'

    # For every image and address, save the image.
    for address, image in zip(addresses, images):
        Image.fromarray(image).save(address, format='PNG')


def ceildiv(a, b):
    """
    Ceiling division to round up the division of a by b.
    """
    return -(a // -b)
