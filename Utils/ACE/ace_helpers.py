""" collection of various helper functions for running ACE"""

from multiprocessing import dummy as multiprocessing
import sys
import os
import copy
import shutil
# from matplotlib import pyplot as plt
# import matplotlib.gridspec as gridspec
# import tcav.model as model
import numpy as np
from PIL import Image
from tqdm import tqdm
# from skimage.segmentation import mark_boundaries
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path

import torchvision

from Utils.model_wrapper import ModelWrapper


class MyModel():
    
    def __init__(self, model, layers):
        
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
            
            self.model = ModelWrapper(model, layers)
        else:
            self.model = ModelWrapper(torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT"), layers)
    
        self.model.eval()
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def run_examples(self, imgs, get_mean=True, bn=None):

        tensor_imgs = torch.from_numpy(imgs).float()

        if len(tensor_imgs.shape) < 4:    
            tensor_imgs = tensor_imgs[None, :]
        
        
        tensor_imgs = tensor_imgs.permute(0, 3, 1, 2)
        
        tensor_imgs = tensor_imgs.to(self.device)

        with torch.no_grad():
            self.model(tensor_imgs)
            acts = self.model.intermediate_activations
            
        if get_mean:
            flattened = {k: torch.flatten(torch.mean(v.detach(), dim=1).cpu(), start_dim=1) for k, v in acts.items()}
        else:
            flattened = {k: torch.flatten(v.detach().cpu(), start_dim=1) for k, v in acts.items()}
            
        output = {k: list(v.numpy()) for k, v in flattened.items()}
        
        return {k: list(v.numpy()) for k, v in flattened.items()}
    
    def label_to_id(self, label):
        default = {"tennis racket": 43}
        return default[label]
    
    def get_gradient(self, imgs, class_id, get_mean=True, return_info=True, test=False):
        
        tensor_imgs = torch.stack(imgs)
        
        del imgs
        
        tensor_imgs = tensor_imgs.to(self.device)

        self.model.to(self.device)
        self.model(tensor_imgs)
        
        del tensor_imgs
        
        gradients, info = self.model.generate_gradients(class_id, test=test)
        
        if get_mean:
            flattened = {k: torch.flatten(torch.mean(v.detach(), dim=1).cpu(), start_dim=1) for k, v in gradients.items()}
        else:
            flattened = {k: torch.flatten(v.detach().cpu(), start_dim=1) for k, v in gradients.items()}
            
        output = {k: list(v.numpy()) for k, v in flattened.items()}
        
        return output, info
    
    def return_grads(self, bns):
        grads = {}
        
        for bn in bns:
            
            body_attribute, layer_attribute, layer_number_attribute, component_atribute = bn.split(".")
            body = getattr(self.model.backbone, body_attribute)
            layer = getattr(body, layer_attribute)
            layer_number = layer[int(layer_number_attribute)]
            component = getattr(layer_number, component_atribute)
            grads[bn] = component.weight.grad.cpu().detach().numpy()

        return grads

def create_directories(output_path, remove_old=True):
    # Create an output directory for our data
    output = Path(output_path)

    # Create the relevant sub-directories.
    discovered_concepts_dir = output / 'concepts/'
    results_dir = output / 'results/'
    cavs_dir = output / 'cavs/'
    activations_dir = output / 'acts/'
    results_summaries_dir = output / 'results_summaries/'
    
    
    # If the directory exists and we want it deleted, delete it.
    if output.exists() and remove_old:
        shutil.rmtree(output)

    # Make all of the directories
    discovered_concepts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    cavs_dir.mkdir(parents=True, exist_ok=True)
    activations_dir.mkdir(parents=True, exist_ok=True)
    results_summaries_dir.mkdir(parents=True, exist_ok=True)

def return_param(param_dict, param_name, default):
    if param_name in param_dict.keys():
        return param_dict[param_name]
    else:
        return default

def load_image_from_file(filename, shape):
    """Given a filename, try to open the file. If failed, return None.
  Args:
    filename: location of the image file
    shape: the shape of the image file to be scaled
  Returns:
    the image if succeeds, None if fails.
  Rasies:
    exception if the image was not the right shape.
  """
    if not Path(filename).exists():
        print('Cannot find file: {}'.format(filename))
        return None
    try:
        img = np.array(Image.open(filename).resize(
            shape, Image.BILINEAR))
        # Normalize pixel values to between 0 and 1.
        img = np.float32(img) / 255.0
        if not (len(img.shape) == 3 and img.shape[2] == 3):
            return None
        else:
            return img

    except Exception as e:
        tf.logging.info(e)
        return None
    return img


def load_images_from_files(filenames, max_imgs=500, return_filenames=False,
                           do_shuffle=True, run_parallel=True,
                           shape=(299, 299),
                           num_workers=100):
    """Return image arrays from filenames.
  Args:
    filenames: locations of image files.
    max_imgs: maximum number of images from filenames.
    return_filenames: return the succeeded filenames or not
    do_shuffle: before getting max_imgs files, shuffle the names or not
    run_parallel: get images in parallel or not
    shape: desired shape of the image
    num_workers: number of workers in parallelization.
  Returns:
    image arrays and succeeded filenames if return_filenames=True.
  """
    imgs = []
    # First shuffle a copy of the filenames.
    filenames = filenames[:]
    if do_shuffle:
        np.random.shuffle(filenames)
    if return_filenames:
        final_filenames = []
    if run_parallel:
        pool = multiprocessing.Pool(num_workers)
        imgs = pool.map(lambda filename: load_image_from_file(filename, shape),
                        filenames[:max_imgs])
        if return_filenames:
            final_filenames = [f for i, f in enumerate(filenames[:max_imgs])
                               if imgs[i] is not None]
        imgs = [img for img in imgs if img is not None]
    else:
        for filename in filenames:
            img = load_image_from_file(filename, shape)
            if img is not None:
                imgs.append(img)
                if return_filenames:
                    final_filenames.append(filename)
            if len(imgs) >= max_imgs:
                break

    if return_filenames:
        return np.array(imgs), final_filenames
    else:
        return np.array(imgs)


def get_acts_from_images(imgs, model, bottleneck_name):
    """Run images in the model to get the activations.
  Args:
    imgs: a list of images
    model: a model instance
    bottleneck_name: bottleneck name to get the activation from
  Returns:
    numpy array of activations.
  """
    return model.run_examples(imgs, bottleneck_name)


def flat_profile(cd, images, bottlenecks=None):
    """Returns concept profile of given images.

  Given a ConceptDiscovery class instance and a set of images, and desired
  bottleneck layers, calculates the profile of each image with all concepts and
  returns a profile vector

  Args:
    cd: The concept discovery class instance
    images: The images for which the concept profile is calculated
    bottlenecks: Bottleck layers where the profile is calculated. If None, cd
      bottlenecks will be used.

  Returns:
    The concepts profile of input images using discovered concepts in
    all bottleneck layers.

  Raises:
    ValueError: If bottlenecks is not in right format.
  """
    profiles = []
    if bottlenecks is None:
        bottlenecks = list(cd.dic.keys())
    if isinstance(bottlenecks, str):
        bottlenecks = [bottlenecks]
    elif not isinstance(bottlenecks, list) and not isinstance(bottlenecks, tuple):
        raise ValueError('Invalid bottlenecks parameter!')
    for bn in bottlenecks:
        profiles.append(cd.find_profile(str(bn), images).reshape((len(images), -1)))
    profile = np.concatenate(profiles, -1)
    return profile


def cross_val(a, b, methods):
    """Performs cross validation for a binary classification task.

  Args:
    a: First class data points as rows
    b: Second class data points as rows
    methods: The sklearn classification models to perform cross-validation on

  Returns:
    The best performing trained binary classification odel
  """
    x, y = binary_dataset(a, b)
    best_acc = 0.
    if isinstance(methods, str):
        methods = [methods]
    best_acc = 0.
    for method in methods:
        temp_acc = 0.
        params = [10 ** e for e in [-4, -3, -2, -1, 0, 1, 2, 3]]
        for param in params:
            clf = give_classifier(method, param)
            acc = cross_val_score(clf, x, y, cv=min(100, max(2, int(len(y) / 10))))
            if np.mean(acc) > temp_acc:
                temp_acc = np.mean(acc)
                best_param = param
        if temp_acc > best_acc:
            best_acc = temp_acc
            final_clf = give_classifier(method, best_param)
    final_clf.fit(x, y)
    return final_clf, best_acc


def give_classifier(method, param):
    """Returns an sklearn classification model.

  Args:
    method: Name of the sklearn classification model
    param: Hyperparameters of the sklearn model

  Returns:
    An untrained sklearn classification model

  Raises:
    ValueError: if the model name is invalid.
  """
    if method == 'logistic':
        return linear_model.LogisticRegression(C=param)
    elif method == 'sgd':
        return linear_model.SGDClassifier(alpha=param)
    else:
        raise ValueError('Invalid model!')


def binary_dataset(pos, neg, balanced=True):
    """Creates a binary dataset given instances of two classes.

  Args:
     pos: Data points of the first class as rows
     neg: Data points of the second class as rows
     balanced: If true, it creates a balanced binary dataset.

  Returns:
    The data points of the created data set as rows and the corresponding labels
  """
    if balanced:
        min_len = min(neg.shape[0], pos.shape[0])
        ridxs = np.random.permutation(np.arange(2 * min_len))
        x = np.concatenate([neg[:min_len], pos[:min_len]], 0)[ridxs]
        y = np.concatenate([np.zeros(min_len), np.ones(min_len)], 0)[ridxs]
    else:
        ridxs = np.random.permutation(np.arange(len(neg) + len(pos)))
        x = np.concatenate([neg, pos], 0)[ridxs]
        y = np.concatenate(
            [np.zeros(neg.shape[0]), np.ones(pos.shape[0])], 0)[ridxs]
    return x, y



def cosine_similarity(a, b):
    """Cosine similarity of two vectors."""
    assert a.shape == b.shape, 'Two vectors must have the same dimensionality'
    a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
    if a_norm * b_norm == 0:
        return 0.
    cos_sim = np.sum(a * b) / (a_norm * b_norm)
    return cos_sim


def similarity(cd, num_random_exp=None, num_workers=25):
    """Returns cosine similarity of all discovered concepts.

  Args:
    cd: The ConceptDiscovery module for discovered conceps.
    num_random_exp: If None, calculates average similarity using all the class's
      random concepts. If a number, uses that many random counterparts.
    num_workers: If greater than 0, runs the function in parallel.

  Returns:
    A similarity dict in the form of {(concept1, concept2):[list of cosine
    similarities]}
  """

    def concepts_similarity(cd, concepts, rnd, bn):
        """Calcualtes the cosine similarity of concept cavs.

    This function calculates the pairwise cosine similarity of all concept cavs
    versus an specific random concept

    Args:
      cd: The ConceptDiscovery instance
      concepts: List of concepts to calculate similarity for
      rnd: a random counterpart
      bn: bottleneck layer the concepts belong to

    Returns:
      A dictionary of cosine similarities in the form of
      {(concept1, concept2): [list of cosine similarities], ...}
    """
        similarity_dic = {}
        for c1 in concepts:
            cav1 = cd.load_cav_direction(c1, rnd, bn)
            for c2 in concepts:
                if (c1, c2) in similarity_dic.keys():
                    continue
                cav2 = cd.load_cav_direction(c2, rnd, bn)
                similarity_dic[(c1, c2)] = cosine_similarity(cav1, cav2)
                similarity_dic[(c2, c1)] = similarity_dic[(c1, c2)]
        return similarity_dic

    similarity_dic = {bn: {} for bn in cd.bottlenecks}
    if num_random_exp is None:
        num_random_exp = cd.num_random_exp
    randoms = ['random500_{}'.format(i) for i in np.arange(num_random_exp)]
    concepts = {}
    for bn in cd.bottlenecks:
        concepts[bn] = [cd.target_class, cd.random_concept] + cd.dic[bn]['concepts']
    for bn in cd.bottlenecks:
        concept_pairs = [(c1, c2) for c1 in concepts[bn] for c2 in concepts[bn]]
        similarity_dic[bn] = {pair: [] for pair in concept_pairs}

        def t_func(rnd):
            return concepts_similarity(cd, concepts[bn], rnd, bn)

        if num_workers:
            pool = multiprocessing.Pool(num_workers)
            sims = pool.map(lambda rnd: t_func(rnd), randoms)
        else:
            sims = [t_func(rnd) for rnd in randoms]
        while sims:
            sim = sims.pop()
            for pair in concept_pairs:
                similarity_dic[bn][pair].append(sim[pair])
    return similarity_dic

def save_concepts(cd, bs=32):
    """Saves discovered concept's images or patches.

  Args:
    cd: The ConceptDiscovery instance the concepts of which we want to save
  """
    for bn in cd.bottlenecks:
        for concept in cd.dic[bn]['concepts']:
            patches_dir = Path(cd.discovered_concepts_dir, bn, concept + '_patches')
            images_dir = Path(cd.discovered_concepts_dir, bn, concept)
            
            for i in range(int(len(cd.dic[bn][concept]['patches']) / bs) + 1):
            
                patches = np.array([np.array(Image.open(img)) for img in cd.dic[bn][concept]['patches'][i * bs:(i + 1) * bs]])
                # patches = (np.clip(loaded_patches, 0, 1) * 256).astype(np.uint8)

                superpixels = np.array([np.array(Image.open(img)) for img in cd.dic[bn][concept]['images'][i * bs:(i + 1) * bs]])
                # superpixels = (np.clip(loaded_superpixels, 0, 1) * 256).astype(np.uint8)

                Path(patches_dir).mkdir(parents=True, exist_ok=True)
                Path(images_dir).mkdir(parents=True, exist_ok=True)

                image_numbers = cd.dic[bn][concept]['image_numbers'][i * bs:(i + 1) * bs]
                image_addresses = [images_dir / f"{img_num}.png" for img_num in image_numbers]
                patch_addresses = [patches_dir / f"{img_num}.png" for img_num in image_numbers]
                
                save_images(patch_addresses, patches)
                save_images(image_addresses, superpixels)


def save_images(addresses, images):
    """Save images in the addresses.

  Args:
    addresses: The list of addresses to save the images as or the address of the
      directory to save all images in. (list or str)
    images: The list of all images in numpy uint8 format.
  """
    if not isinstance(addresses, list):
        image_addresses = []
        for i, image in enumerate(images):
            image_name = '0' * (3 - int(np.log10(i + 1))) + str(i + 1) + '.png'
            image_addresses.append(os.path.join(addresses, image_name))
        addresses = image_addresses
    assert len(addresses) == len(images), 'Invalid number of addresses'
    for address, image in zip(addresses, images):
        Image.fromarray(image).save(address, format='PNG')

def ceildiv(a, b):
    return -(a // -b)
