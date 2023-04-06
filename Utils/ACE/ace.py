"""ACE script.

Script for discovering and testing concept activation vectors. It contains the
ConceptDiscovery class that aims to discover potential concepts for the target class.
It then generates CAVs for these potential concepts and performs tests to determine
how meaningful they are as well as their statistical significance.
"""

import sys
import os
from pathlib import Path
import pickle
import shutil
import random
import math
from time import sleep

import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import skimage.segmentation as segmentation
from skimage.segmentation import mark_boundaries
from sklearn.decomposition import IncrementalPCA
import sklearn.cluster as cluster
import sklearn.metrics.pairwise as metrics
import torch
import torchvision.transforms as T

from Utils.ACE.ace_helpers import *
from Utils.TCAV import cav


class ConceptDiscovery(object):
    """
    Class for discovering and testing potential concepts for a class.

    For a trained network, it first discovers the concepts as areas of the images
    in the class and then calculates the TCAV score of each concept. It is also
    able to transform images from pixel space into concept space.
    """

    def __init__(self, model, target_class, source_dir, output_dir, bottlenecks, num_random_exp=2,
                 channel_mean=True, min_imgs=20, average_image_value=117, resize_dims=(512, 512),
                 pca_n_components=1000):
        """

        :param model: A trained classification model on which we run the concept discovery algorithm.
        :param target_class: Name of the one of the classes of the network.
        :param bottlenecks: A list of bottleneck layers of the model for which the concept discovery stage is performed.
        :param source_dir: This directory that contains folders with images of network's classes.
        :param output_dir: Directory to save output to.
        :param num_random_exp: Number of random counterparts used for calculating several CAVs and TCAVs for each
        concept (to make statistical testing possible).
        :param channel_mean: If true, for the unsupervised concept discovery the bottleneck activations are averaged
        over channels instead of using the whole acivation vector (reducing dimensionality).
        :param min_imgs: Minimum number of images in a discovered concept for the concept to be accepted.
        :param average_image_value: The average value used for mean subtraction in the network's preprocessing stage.
        :param resize_dims: A tuple defining the height and weight to resize images to.
        """

        # Save the model and target class to an instance variable.
        self.model = model
        self.target_class = target_class

        # Save the number of random experiments and the random concept.
        self.num_random_exp = num_random_exp
        self.random_concept = "Random_concept"

        # Save the bottlenecks
        self.bottlenecks = bottlenecks

        # Save the directories we will need to access
        self.source_dir = Path(source_dir)
        
        self.output_dir = Path(output_dir)
        self.discovered_concepts_dir = self.output_dir / 'concepts/'
        self.results_dir = self.output_dir / 'results/'
        self.cav_dir = self.output_dir / 'cavs/'
        self.activation_dir = self.output_dir / 'acts/'
        self.results_summaries_dir = self.output_dir / 'results_summaries/'

        # Save channel mean option.
        if type(channel_mean) != list:
            self.channel_mean = len(bottlenecks) * [channel_mean]
        else:
            self.channel_mean = channel_mean

        # Save image details
        #self.max_imgs = max_imgs
        self.min_imgs = min_imgs

        # Save the average image value.
        self.average_image_value = average_image_value

        # Save the tuple that defines the dimensions to resize images to.
        self.resize_dims = resize_dims
        
        self.pca = None
            
        # Variable for saving instances of PCA for converting activations and gradients to the new space.
        pca_file_path = self.activation_dir / "PCA.pkl"
        if pca_file_path.is_file():
            
            # Open the activation file and read it in.
            with open(pca_file_path, 'rb') as handle:
                self.pca = pickle.load(handle)
        
        if type(pca_n_components) != list:
            self.pca_n_components = len(bottlenecks) * [pca_n_components]
        else:
            self.pca_n_components = pca_n_components
    
    
    def initialize_random_concept_and_samples(self):
        """
        This function create a folder for a random concept and creates random samples of
        from all of the superpixels for training the discovered concepts against.
        """
        
        # Get the list of superpixels.
        superpixels = self.discovered_concepts_dir / "superpixels"
        list_of_files = list(superpixels.iterdir())
        
        # Create random selection for random concept.
        random.seed(42)
        random_concept_superpixels = np.array(random.sample(list_of_files, self.min_imgs))
        
        # Save these images to a subfolder called Concept in Random.
        destination = self.discovered_concepts_dir / "Random" / self.random_concept / "superpixels"
        destination.mkdir(parents=True, exist_ok=True)
        
        # Save the random concept images.
        for img in random_concept_superpixels:
            shutil.copy(img, destination / img.name)
        
        # Get the corresponding patches
        random_concept_patches = [img.parent.parent / "patches" / img.name for img in random_concept_superpixels]
        
                
        # Save these images to a subfolder called Concept in Random.
        destination = self.discovered_concepts_dir / "Random" / self.random_concept / "patches"
        destination.mkdir(parents=True, exist_ok=True)
        
        # Save the random concept images.
        for img in random_concept_patches:
            shutil.copy(img, destination / img.name)
            
        # For every random experiment that we need a random sample for.
        for i in range(self.num_random_exp):
            
            # Number this sample.
            random_num = f"Random_{i:03d}"
            
            # Get the random sample.
            random_sample_imgs =  np.array(random.sample(list_of_files, self.min_imgs))
        
            # Create the directory to store the images.
            destination = self.discovered_concepts_dir / "Random" / random_num
            destination.mkdir(parents=True, exist_ok=True)
                
            # Save the random sample.
            for img in random_sample_imgs:

                shutil.copy(img, destination / img.name)
    
#     def load_concept_imgs(self, concept, max_imgs=1000):
#         """
#         This function loads images for the given concept from the source directory.

#         :param concept: The name of the concept to be loaded.
#         :param max_imgs: Maximum number of images to be loaded.
#         :return: Images of the desired concept or class.
#         """

#         # Define the directory to extract the concept images from.
#         concept_dir = self.source_dir / concept / "discovery"

#         # Form a list of the image paths.
#         img_paths = [
#             os.path.join(concept_dir, d)
#             for d in concept_dir.iterdir()
#         ]

#         # Return the output from this function
#         return load_images_from_files(
#             img_paths,
#             max_imgs=max_imgs,
#             return_filenames=False,
#             do_shuffle=False,
#             shape=self.resize_dims,
#             run_parallel=False,
#             num_workers=0)

    def create_patches(self, method='slic', discovery_images=None,
                       param_dict=None):
        """Creates a set of image patches using superpixel methods.

        This method takes in the concept discovery images and transforms it to a
        dataset made of the patches of those images.

        :param method: The superpixel method used for creating image patches. One of
        'slic', 'watershed', 'quickshift', 'felzenszwalb'.
        :param discovery_images: Images used for creating patches. If None, the images in
        the target class folder are used.
        :param param_dict: Contains parameters of the superpixel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
                method.
        """

        # Specify the directory to save the superpixels and patches.
        superpixel_dir = self.discovered_concepts_dir / "superpixels"
        patch_dir = self.discovered_concepts_dir / "patches"

        # Make these directories
        superpixel_dir.mkdir(exist_ok=True, parents=True)
        patch_dir.mkdir(exist_ok=True, parents=True)

        # Create an empty param dict if we have not specified one.
        if param_dict is None:
            param_dict = {}

        # Create empty lists for storing the superpixels, image numbers and patches.
        dataset, image_numbers, patches = [], [], []

        # If we have not specified discovery images, we must load them.
        if discovery_images is None:

            # Get the list of discovery image paths.
            concept_dir = self.source_dir / self.target_class / "discovery"
            
            # Save the list of discovery images paths
            self.discovery_images = list(concept_dir.iterdir())
            

        # Otherwise, we can use the ones that have been supplied.
        else:
            self.discovery_images = discovery_images

        # For every image in the set.
        for image_num, img_path in enumerate(tqdm(self.discovery_images, total=len(self.discovery_images))):
            # Get the superpixels for this image using the given method.
            img = load_image_from_file(img_path, self.resize_dims)
            
            channel_axis = next(iter([i for i in range(len(img.shape)) if img.shape[i] == 3]))

            image_superpixels, image_patches = self._return_superpixels(
                img, method, channel_axis, param_dict)

            # Convert both of the outputs to numpy arrays.
            superpixels, patches = np.array(image_superpixels), np.array(image_patches)

            # Convert both to int8 type.
#             superpixels = (np.clip(superpixels, 0, 1) * 256).astype(np.uint8)
#             patches = (np.clip(patches, 0, 1) * 256).astype(np.uint8)

            # Generate addresses to save the superpixels and patches under.
            superpixel_addresses = [superpixel_dir / f"{image_num:03d}_{i:03d}.png" for i in range(len(image_superpixels))]
            patch_addresses = [patch_dir / f"{image_num:03d}_{i:03d}.png" for i in range(len(image_superpixels))]

            # Save both the superpixels and patches to the generated addresses.
            save_images(superpixel_addresses, superpixels)
            save_images(patch_addresses, patches)

    def _return_superpixels(self, img, method='slic', channel_axis=2, param_dict=None):
        """Returns all patches for one image.

        Given an image, calculates superpixels for each of the parameter lists in
        param_dict and returns a set of unique superpixels by
        removing duplicates. If two patches have Jaccard similarity more than 0.5,
        they are considered duplicates.

        :param img: The input image. :param method: Superpixel method, one of slic, watershed, quichsift,
        or felzenszwalb. :param param_dict: Contains parameters of the superpixel method used in the form of {
        'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance {'n_segments':[15,50,80], 'compactness':[10,10,
        10]} for slicmethod.
        :return: The generated superpixels and patches.
        """

        # If a param dict has not been supplied, create an empty one.
        if param_dict is None:
            param_dict = {}

        # If the method is slic, pull the params out, or use the default if they weren't specified.
        if method == 'slic':
            
            # Return the parameter for n_segments if it is in the dict, otherwise return the default.
            n_segmentss = return_param(param_dict, "n_segments", [15, 50, 80])

            n_params = len(n_segmentss)
            
            compactnesses = return_param(param_dict, "compactness", [20] * n_params)
            sigmas = return_param(param_dict, "sigma", [1.] * n_params)

        # If the method is watershed, pull the params out or use the default.
        elif method == 'watershed':
            markerss = return_param(param_dict, "marker", [15, 50, 80])

            n_params = len(markerss)
            
            compactnesses = return_param(param_dict, "compactness", [0.] * n_params)
          
        # If the method is quickshift, pull the params out or use the default.
        elif method == 'quickshift':
            max_dists = return_param(param_dict, "max_dist", [20, 15, 10])

            n_params = len(max_dists)
            ratios = return_param(param_dict, "ratio", [1.0] * n_params)
            kernel_sizes = return_param(param_dict, "kernel_size", [10] * n_params)

        # If the method is felzenszwalb, pull the params out or use the default.
        elif method == 'felzenszwalb':
            scales = return_param(param_dict, "scale", [1200, 500, 250])

            n_params = len(scales)
            
            sigmas = return_param(param_dict, "sigma", [0.8] * n_params)
            min_sizes = return_param(param_dict, "min_size", [20] * n_params)

        # Otherwise, the method provided is not supported.
        else:
            # Raise an error.
            raise ValueError('Invalid superpixel method!')

        # Create a list for storing the unique masks.
        unique_masks = []

        # For every combination of parameters
        for i in range(n_params):

            # List for storing these masks.
            param_masks = []

            # Logic for using the correct segmentation.
            if method == 'slic':
                segments = segmentation.slic(
                    img, n_segments=n_segmentss[i], compactness=compactnesses[i],
                    sigma=sigmas[i], channel_axis=channel_axis)
            elif method == 'watershed':
                segments = segmentation.watershed(
                    img, markers=markerss[i], compactness=compactnesses[i], channel_axis=channel_axis)
            elif method == 'quickshift':
                segments = segmentation.quickshift(
                    img, kernel_size=kernel_sizes[i], max_dist=max_dists[i],
                    ratio=ratios[i], channel_axis=channel_axis)
            elif method == 'felzenszwalb':
                segments = segmentation.felzenszwalb(
                    img, scale=scales[i], sigma=sigmas[i], min_size=min_sizes[i], channel_axis=channel_axis)

            # Iterate through all of the segmentation masks up until the max value.
            for s in range(segments.max()):
                mask = (segments == s).astype(float)

                # If the mask is meaningful and has values, proceed.
                if np.mean(mask) > 0.001:
                    unique = True

                    # Determine if this mask is unique.
                    for seen_mask in unique_masks:
                        jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                        if jaccard > 0.5:
                            unique = False
                            break

                    # If it is unique, add it to our list.
                    if unique:
                        param_masks.append(mask)

            # Add the current masks to the total unqiue.
            unique_masks.extend(param_masks)

        # Lists for storing the superpixels and patches.
        superpixels, patches = [], []

        # While unique masks has values, run this loop.
        while unique_masks:
            # Extract the superpixels and patch for the next mask in the list.
            superpixel, patch = self._extract_patch(img, unique_masks.pop())

            # Add the values to the superpixel and patch list respectively.
            superpixels.append(superpixel)
            patches.append(patch)

        # Return the superpixels and patches.
        return superpixels, patches

    def _extract_patch(self, image, mask):
        """Extracts a patch out of an image.

        :param image: The original image.
        :param mask: The binary mask of the patch area.
        :return: Superpixel and patch.
        """
        
        ones = (mask == 1.)
        
        mask_expanded = ones[...,np.newaxis]
        
        patch = image * mask_expanded
        
#         patch = (mask_expanded * image).astype(np.uint8) #  + (1 - mask_expanded) * float(self.average_image_value) / 255)
        ones = np.where(mask == 1)
        h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
        
        image = Image.fromarray(patch[h1:h2, w1:w2])

        # Resize the image and convert it to a float.
        image_resized = np.array(image.resize(self.resize_dims, Image.BICUBIC))

        return image_resized, patch

    def _get_activations(self, img_paths, paths=True, bs=2):
        """Returns activations of a list of imgs.

        :param img_paths: List of paths to the images to get activations for.
        :param paths: True if the list is paths, if False the list will be assumed to contain image data.
        :param bs: Batch size to be used.
        :param channel_mean: Should the activations be averaged across the channels/filters. Reduces the complexity at
        the cost of accuracy.
        :return: A dictionary with the keys as the supplied bottleneck layers with the activations as the values.
        """

        # Create a list to store the output.
        output = []
        acts_processed = 0
        
        if not all(self.channel_mean) and self.pca == None:
            
            print("The activations are being calculated and then PCA will be computed on the activations to lower the dimensionality. This will take some time.")
            
            self.pca = {}
            
            activations_path = self.output_dir / "acts/"
#             superpixel_activation_path = activations_path / "superpixels/"
            
            # Loop through all the image paths taking the batch size each time.
            for i in tqdm(range(ceildiv(img_paths.shape[0], bs)), total=ceildiv(img_paths.shape[0], bs),
                          desc="Calculating activations for PCA"):
                
                batch_path = activations_path / f"{i}_acts.pkl"
                
                if batch_path.is_file():
                    continue

                # Load the images we need if the paths are supplied.
                if paths:
                    # For every image in the batch, open the image and convert it to a numpy array.
                    imgs = [np.array(Image.open(img)) for img in img_paths[i * bs:(i + 1) * bs]]

                # Otherwise we can just use the passed images.
                else:
                    imgs = img_paths

                # Append the returned activations from running the model.
                activations = self.model.run_examples(np.array(imgs), self.channel_mean)
                
                with open(batch_path, 'wb') as handle:
                    pickle.dump(activations, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                del activations
            
            
            
            # Fit the IncrementalPCA for each bottleneck, load them individually so we can save memory.
            for bn, pca_n_components in zip(self.bottlenecks, self.pca_n_components):
                
                self.update_pca(bn, pca_n_components, img_paths, bs, activations_path)
                 
            # Read in each activation file
            for act_path in activations_path.iterdir():
                
                # Open the activation file and read it in.
                with open(act_path, 'rb') as handle:
                    activations = pickle.load(handle)
                
                # Convert the values in each dimension
                for bn in self.bottlenecks:
                    activations[bn] = self.pca[bn].transform(activations[bn])
            
                # Save the activation output.
                output.append(activations)
            

        else:
            # Loop through all the image paths taking the batch size each time.
            for i in tqdm(range(ceildiv(img_paths.shape[0], bs)), total=ceildiv(img_paths.shape[0], bs),
                          desc="Calculating activations for superpixels"):

                # Load the images we need if the paths are supplied.
                if paths:
                    # For every image in the batch, open the image and convert it to a numpy array.
                    imgs = [np.array(Image.open(img)) for img in img_paths[i * bs:(i + 1) * bs]]

                # Otherwise we can just use the passed images.
                else:
                    imgs = img_paths

                activations = self.model.run_examples(np.array(imgs), self.channel_mean)

                if not self.channel_mean:
                    for bn in self.bottlenecks:
                        activations[bn] = self.pca[bn].transform(activations[bn])

                output.append(activations)


        # Dict to store the activations.
        aggregated_out = {}

        # For every layer.
        for k in output[0].keys():
            # Take all the batch outputs for that layer and concatenate the results.
            aggregated_out[k] = np.concatenate(list(d[k] for d in output))

        return aggregated_out
    
    def update_pca(self, bn, pca_n_components, img_paths, bs, activations_path)

        # Find the number of batches in each to allow for n_components
        # Use divmod to find the number of complete batches, then split the remainder across the batches.
        num_batches, remainder = divmod(img_paths.shape[0], pca_n_components)

        # Use the floor rounding so we make sure the last batch always has enough. If we round up and take too much in the
        # first batches we may be left short
        activation_dicts_per_batches = (pca_n_components / bs) + math.floor(remainder / bs / num_batches)

        self.pca[bn] = IncrementalPCA(n_components=pca_n_components, copy=False)

        pca_batches_complete = 0
        current_pca_batch = []

        for batch in activations_path.iterdir():

            # Open the activation file and read it in.
            with open(batch, 'rb') as handle:
                activations = pickle.load(handle)

            current_pca_batch.append(activations[bn])

            del activations


            if pca_batches_complete < num_batches and len(current_pca_batch) == activation_dicts_per_batches:

                pca_batches_complete += 1

                # Take all the batch outputs for that layer and concatenate the results.
                aggregated_acts = np.concatenate(current_pca_batch)
                print(aggregated_acts.shape)

                sleep(5)

                self.pca[bn].partial_fit(aggregated_acts, check_input=False)
                print("Complete PCA run")

                sleep(10)

                del aggregated_acts
                current_pca_batch.clear()

        sleep(5)


        if len(current_pca_batch) > 0:
            # Run the final batch through
            aggregated_acts = np.concatenate(current_pca_batch)

            self.pca[bn].partial_fit(aggregated_acts)


        with open(activations_path / "PCA.pkl", 'wb') as handle:
            pickle.dump(self.pca, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _cluster(self, acts, method='KM', param_dict=None):
        """Runs unsupervised clustering algorithm on concept actiavtations.

        :param acts: activation vectors of datapoints points in the bottleneck layer.
        E.g. (number of clusters,) for Kmeans
        :param method: clustering method. We have:
        'KM': Kmeans Clustering
        'AP': Affinity Propagation
        'SC': Spectral Clustering
        'MS': Mean Shift clustering
        'DB': DBSCAN clustering method
        :param param_dict: Contains superpixel method's parameters. If an empty dict is
                 given, default parameters are used.
        :return:
        asg: The cluster assignment label of each data points
        cost: The clustering cost of each data point
        centers: The cluster centers. For methods like Affinity Propagetion
        where they do not return a cluster center or a clustering cost, it
        calculates the medoid as the center  and returns distance to center as
        each data points clustering cost.
        """

        # Create an empty param dict if we don't have one.
        if param_dict is None:
            param_dict = {}

        # Initialize the centres as None
        centers = None

        if method == 'KM':
            n_clusters = return_param(param_dict, "n_clusters", 25)

            km = cluster.KMeans(n_clusters)
            d = km.fit(acts)

            centers = km.cluster_centers_

            d = np.linalg.norm(
                np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
            asg, cost = np.argmin(d, -1), np.min(d, -1)

        elif method == 'AP':
            damping = return_param(param_dict, "damping", 0.5)

            ca = cluster.AffinityPropagation(damping)
            ca.fit(acts)

            centers = ca.cluster_centers_

            d = np.linalg.norm(
                np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
            asg, cost = np.argmin(d, -1), np.min(d, -1)

        elif method == 'MS':
            ms = cluster.MeanShift()
            asg = ms.fit_predict(acts)

        elif method == 'SC':
            n_clusters = return_param(param_dict, "n_clusters", 25)

            sc = cluster.SpectralClustering(n_clusters=n_clusters)
            asg = sc.fit_predict(acts)

        elif method == 'DB':
            eps = return_param(param_dict, "eps", 0.5)
            min_samples = return_param(param_dict, "min_samples", 20)

            sc = cluster.DBSCAN(eps, min_samples, n_jobs=self.num_workers)
            asg = sc.fit_predict(acts)

        else:
            raise ValueError('Invalid Clustering Method!')

        # If clustering returned cluster centers, use medoids
        if centers is None:

            centers = np.zeros((asg.max() + 1, acts.shape[1]))
            cost = np.zeros(len(acts))

            for cluster_label in range(asg.max() + 1):
                cluster_idxs = np.where(asg == cluster_label)[0]
                cluster_points = acts[cluster_idxs]

                pw_distances = metrics.euclidean_distances(cluster_points)

                centers[cluster_label] = cluster_points[np.argmin(
                    np.sum(pw_distances, -1))]

                cost[cluster_idxs] = np.linalg.norm(
                    acts[cluster_idxs] - np.expand_dims(centers[cluster_label], 0),
                    ord=2,
                    axis=-1)

        return asg, cost, centers

    def discovery_images_size(self, target_class):
        """
        This functionr returns the number of images in the target class for discovery.

        :param target_class: The class we want to find the number of images for.
        :param num_discovery_imgs: The max number of discovery images.
        :return: The count of images or max size allowed, whichever is smaller.
        """

        # Find the discovery directory.
        discovery_dir = self.source_dir / target_class / "discovery"

        # Get the images by iterating the directory.
        discovery_images = np.array(list(discovery_dir.iterdir()))

        # Return the smallest of the images in the directory or the max allowed.
        return len(discovery_images)

    def discover_concepts(self, method='KM', activations=None, param_dicts=None, bs=2):
        """Discovers the frequent occurring concepts in the target class.

        Calculates self.dic, a dictionary containing all the information of the
        discovered concepts in the form of {'bottleneck layer name: bn_dic} where
        bn_dic itself is in the form of {'concepts:list of concepts,
        'concept name': concept_dic} where the concept_dic is in the form of
        {'images': resized patches of concept, 'patches': original patches of the
        concepts, 'image_numbers': image id of each patch}

        :param method: Clustering method.
        :param activations: If activations are already calculated. If not calculates
                   them. Must be a dictionary in the form of {'bn':array, ...}
        :param param_dicts: A dictionary in the format of {'bottleneck':param_dict,...}
                   where param_dict contains the clustering method's parametrs
                   in the form of {'param1':value, ...}. For instance for Kmeans
                   {'n_clusters':25}. param_dicts can also be in the format
                   of param_dict where same parameters are used for all
                   bottlenecks.
        """

        # If a param dict is not specified use an empty one.
        if param_dicts is None:
            param_dicts = {}
        # Make sure that the param dict has the bottleneck layers as keys with params for each.
        if set(param_dicts.keys()) != set(self.bottlenecks):
            param_dicts = {bn: param_dicts for bn in self.bottlenecks}

        # The main dictionary of the ConceptDiscovery class.
        self.dic = {}

        # Get the discovery size.
        discovery_size = self.discovery_images_size(self.target_class)

        # If we don't have any or are missing activations, get them.
        if activations is None or set(self.bottlenecks) != set(activations.keys()):
            # Get the superpixel images.
            superpixels_dir = self.discovered_concepts_dir / "superpixels"
            superpixel_images = np.array(list(superpixels_dir.iterdir()))

            # Get the patch images.
            patches_dir = self.discovered_concepts_dir / "patches"
            patch_images = np.array(list(patches_dir.iterdir()))

            # Get the activations back after passing the superpixels.
            activations = self._get_activations(superpixel_images, bs=bs)

        # For every bottleneck we will cluster.
        for bn in self.bottlenecks:

            # Dictionary to store results and get the activations.
            bn_dic = {}
            bn_activations = activations[bn]

            # Cluster the activations
            bn_dic['label'], bn_dic['cost'], centers = self._cluster(bn_activations, method, param_dicts[bn])

            # Set the concept number and create a list under "concepts" in the bn_dic.
            concept_number, bn_dic['concepts'] = 0, []

            # For every cluster label returned.
            for i in range(bn_dic['label'].max() + 1):

                # Get the indexes with that label.
                label_idxs = np.where(bn_dic['label'] == i)[0]

                # If we pass the minimum number of images for a concept.
                if len(label_idxs) > self.min_imgs:

                    # Add the details for this cluster to the dic for the current bottleneck layer.
                    concept_costs = bn_dic['cost'][label_idxs]
                    concept_idxs = label_idxs[np.argsort(concept_costs)]
                    concept_image_numbers = set([int(p.name.split("_")[0]) for p in patch_images[label_idxs]])
                    highly_common_concept = len(
                        concept_image_numbers) > 0.5 * len(label_idxs)
                    mildly_common_concept = len(
                        concept_image_numbers) > 0.25 * len(label_idxs)
                    mildly_populated_concept = len(
                        concept_image_numbers) > 0.25 * discovery_size
                    cond2 = mildly_populated_concept and mildly_common_concept
                    non_common_concept = len(
                        concept_image_numbers) > 0.1 * len(label_idxs)
                    highly_populated_concept = len(
                        concept_image_numbers) > 0.5 * discovery_size
                    cond3 = non_common_concept and highly_populated_concept
                    if highly_common_concept or cond2 or cond3:
                        concept_number += 1
                        concept = '{}_concept{}'.format(self.target_class, concept_number)
                        bn_dic['concepts'].append(concept)
                        bn_dic[concept] = {
                            'images': superpixel_images[concept_idxs],
                            'patches': patch_images[concept_idxs],
                            'image_numbers': [str(p.name.split(".")[0]) for p in patch_images[concept_idxs]]
                        }
                        bn_dic[concept + '_center'] = centers[i]

            # Remove the label and cost from the dictionary.
            bn_dic.pop('label', None)
            bn_dic.pop('cost', None)

            # Save the concept details for this layer into the overall dict.
            self.dic[bn] = bn_dic

        # Save the concept dict so we don't need to recompute it later.
        self.save_concept_dict()

    def save_concept_dict(self):
        """
        This function saves the concept dictionary into a pickle file, so it can be reloaded if the processes is
        interruprted.
        """

        # Open a .pkl file in the concept directory and save the concept dictionary to it.
        with open(self.cav_dir / 'concept_dict.pkl', 'wb') as handle:
            pickle.dump(self.dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_concept_dict(self):
        """
        This function loads a saved concept dictionary for use.
        """

        # Open the concept dictionary file and save it to self.dic
        with open(self.cav_dir / 'concept_dict.pkl', 'rb') as handle:
            self.dic = pickle.load(handle)

    def _calculate_cav(self, c, r, bn, act_c, act_r, ow):
        """
        Calculates a sinle cav for a concept and a one random counterpart.
        
        :param c: Concept name.
        :param r: Random concept name.
        :param bn: The bottleneck layer name.
        :param act_c: Activation matrix of the concept in the 'bn' layer
        :param act_r: Random activation matrix.
        :param ow: overwrite if CAV already exists
        :return: The accuracy of the CAV
        """

        # Train or load CAV by passing the concept details and the activations.
        cav_instance = cav.load_or_train_cav([c, r], bn, self.cav_dir,
                                             activations={c: {bn: act_c}, r: {bn: act_r}},
                                             overwrite=ow)

        # Return the CAV accuracy.
        return cav_instance.accuracies['overall']

    def _concept_cavs(self, bn, concept, random, activations, random_activations, randoms=None, ow=True):
        """
        Calculates CAVs of a concept versus all the random counterparts.
        
        :param bn: bottleneck layer name
        :param concept: the concept name
        :param activations: activations of the concept in the bottleneck layer
        :param random_activations: 
        :param randoms: None if the class random concepts are going to be used
        :param ow: If true, overwrites the existing CAVs
        :return: A dict of cav accuracies in the form of {'bottleneck layer':
      {'concept name':[list of accuracies], ...}, ...}
        """

        accs = self._calculate_cav(concept, random, bn, activations, random_activations, ow)
        return accs

    def cavs(self, min_acc=0, ow=True, bs=2):
        """
        Calculates cavs for all discovered concepts.

        This method calculates and saves CAVs for all the discovered concepts
        versus all random concepts in all the bottleneck layers
        
        :param min_acc: Delete discovered concept if the average classification accuracy of the CAV is less than min_acc
        :param ow: If True, overwrites an already calculated cav.
        :param bs: The batch size for calculating the activations.
        :return: The accuracies of all the CAVs generated.
        """

        # Create a dictionary for the accuracies and a list to track concepts to delete.
        acc = {bn: {} for bn in self.bottlenecks}
        concepts_to_delete = []

        # If we don't have the concept dic, load it.
        if not hasattr(self, "dic"):
            self.load_concept_dict()
        
        # Get the Random directory.
        random_dir = self.discovered_concepts_dir / "Random"
        
        # Get the images for the random concept.
        random_concept_imgs = np.array(list((random_dir / self.random_concept / "superpixels").iterdir()))
        
        # Get the activations for the random concept.
        rnd_concept_acts = self._get_activations(random_concept_imgs)
        
        # Create a dictionary to store the random concepts
        all_random_acts = {}
        
        # Get the rest of the random samples
        for directory in random_dir.iterdir():
            
            # If we are at the Concept directory, skip it.
            if directory.name == self.random_concept:
                continue
            
            # Get a numpy array of the images.
            imgs = np.array(list(directory.iterdir()))
            
            # Get the activations for this random sample.
            sample_activations = self._get_activations(imgs, bs=bs)
            
            # Add the current random sample to the 
            all_random_acts[directory.name] = sample_activations
        
        # Dictionary for storung concept activations for use between layers.
        concept_acts_dict = {}
        
        # For every bottleneck.
        for bn in self.bottlenecks:
            
            def random_helper(random, random_acts):
                return self._concept_cavs(bn, self.random_concept, random, rnd_concept_acts[bn], random_acts[bn], ow=ow)
            
            # Compute the random concept accuracy.
            acc[bn][self.random_concept] = [random_helper(k, v) for k, v in all_random_acts.items()]

            # For every concept
            for concept in self.dic[bn]['concepts']:

                # Get the images for the concept.
                concept_imgs = self.dic[bn][concept]['images']

                # If we have yet to get the activations for this concept, get them now.
                if concept not in concept_acts_dict.keys():
                    concept_acts_dict[concept] = self._get_activations(concept_imgs, bs=bs)
                
                # Extract the activations for the current concept in the current bottlneck layer.
                concept_acts = concept_acts_dict[concept][bn]
            
                # Define a function to accept the random concept and the random activations.
                def concept_helper(random, random_acts):
                    return self._concept_cavs(bn, concept, random, concept_acts, random_acts[bn], ow=ow)
                
                # Add the list of accuracies for the concept to the dictionary.
                acc[bn][concept] = [concept_helper(k, v) for k, v in all_random_acts.items()]
                
                # If the mean of the CAV accuracies is less than the min, delete the concept.
                if np.mean(acc[bn][concept]) < min_acc:
                    concepts_to_delete.append((bn, concept))

        # Delete the concept if it is not accurate enough.
        for bn, concept in concepts_to_delete:
            self.delete_concept(bn, concept)
        
        # Open a .pkl file in the concept directory and save the concept accuracies to it.
        with open(self.cav_dir / 'concept_accuracies.pkl', 'wb') as handle:
            pickle.dump(acc, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return acc

    def load_cav_direction(self, concept, random, bn, directory=None):
        """
        Loads an already computed CAV.
        
        :param concept: Concept name.
        :param random: Random concept name.
        :param bn: Bottleneck layer.
        :param directory: Where CAV is saved.
        
        :return: The CAV instance.
        """

        # If a directory is not specified, use self.cav_dir
        if directory is None:
            directory = self.cav_dir

        # Load the CAV.
        loaded_cav = cav.load_or_train_cav([concept, random], bn, directory)
        
        if loaded_cav is None:
            print(f"Concept: {concept}")
            print(f"Random: {random}")
            print(f"Bottle: {bn}")
            print(f"directory: {directory}")
        
        # Extract the vector from the CAV.
        vector = loaded_cav.get_cav(concept)

        return vector

    def _sort_concepts(self, scores):
        for bn in self.bottlenecks:
            tcavs = []
            for concept in self.dic[bn]['concepts']:
                tcavs.append(np.mean(scores[bn][concept]))
            concepts = []
            for idx in np.argsort(tcavs)[::-1]:
                concepts.append(self.dic[bn]['concepts'][idx])
            self.dic[bn]['concepts'] = concepts

    def _return_gradients(self, images, paths=True, test=False):
        """
        For the given images, calculate a dictionary of gradients for each layer.
        The corresponding images and detection info is returned.
        
        :param images: Images for which we want to calculate gradients.
        :param paths: Whether the supplied list contains paths, if False the list must contain images.
        
        :return: Dictionary of gradients and info on which image and bounding box they came from.
        """

        # Initialize variables to store the gradients and info.
        gradients = {k: [] for k in self.bottlenecks}
        total_info = []

        # Get the class id for the label we have.
        class_id = self.model.label_to_id(self.target_class.replace('_', ' '))

        # Loop through all the images, one at a time.
        for i in tqdm(range(len(images)), total=len(images), desc="Calculating gradients"):

            # Load the image we need
            if paths:
                img = [T.ToTensor()(Image.open(images[i]).resize(self.resize_dims, Image.BICUBIC))]
            else:
                img = images[i]

            # Pass the image to the get_gradient method and capture the returned gradients and corresponding info.
            img_gradients, detection_info = self.model.get_gradient(img, class_id, self.channel_mean, test=test)

            del img

            # Add the information regarding the current info to the detection info.
            current_info = [f"{images[i].name}_{part}" for part in detection_info]

            # Add this info to the total, so we can correspond them to the gradients.
            total_info = total_info + current_info

            # Iterate through the layers we have and add the corresponding gradients to our total for the layer.
            for layer, vals in img_gradients.items():
                
                #
                if self.channel_mean:
                    final_vals = self.pca[layer].transform(vals)
                else:
                    final_vals = vals
                
                # Add these gradients to the total we have collected so far
                gradients[layer].append(final_vals)

#         #Convert the lists to numpy arrays
#         for k, v in gradients.items():
#             gradients[k] = np.vstack(v)

        return gradients, total_info

    def _tcav_score(self, bn, concept, rnd, gradients):
        """
        Calculates and returns the TCAV score of a concept.
        
        :param bn: bottleneck layer.
        :param concept: Concept name.
        :param rnd: Random counterpart.
        :param gradients: Dict of gradients of tcav_score_images.
        
        :return: TCAV score of the concept with respect to the given random counterpart
        """
        
        # Get the CAV vector.
        vector = self.load_cav_direction(concept, rnd, bn)
        
        # Multiply the CAV vector with the gradients
        prod = np.sum(gradients[bn] * vector, -1)
        
        return np.mean(prod < 0)

    def tcavs(self, test=False, sort=True, tcav_score_images=None):
        """
        Calculates TCAV scores for all discovered concepts and sorts concepts.

        This method calculates TCAV scores of all the discovered concepts for
        the target class using all the calculated CAVs. It later sorts concepts
        based on their TCAV scores.
        
        :param test: If true, perform statistical testing and removes concepts that don't pass
        :param sort: If true, it will sort concepts in each bottleneck layers based on average TCAV score of the
        concept.
        :param tcav_score_images: Target class images used for calculating tcav scores.
        If None, the target class source directory images are used.
        
        :return: A dictionary of the form {'bottleneck layer':{'concept name':
        [list of tcav scores], ...}, ...} containing TCAV scores.
        """
        
        # Initialize a dictionary to store the scores.
        tcav_scores = {bn: {} for bn in self.bottlenecks}

        # If we don't have the concept dictionary, load it in.
        if not hasattr(self, "dic"):
            self.load_concept_dict()

        random_samples = [f"Random_{i:03d}" for i in range(self.num_random_exp)]

        # If we have not got tcav score images, load the images from the source directory.
        if tcav_score_images is None:  # Load target class images if not given
            files = self.source_dir / self.target_class / "tcav"
            tcav_score_images = list(files.iterdir())

        # Accept image paths from target class folder?
        gradients, _ = self._return_gradients(tcav_score_images)
        

        # For every bottleneck and concept
        for bn in self.bottlenecks:
            
            for concept in self.dic[bn]['concepts'] + [self.random_concept]:
                
                def test_function(rnd):
                    return self._tcav_score(bn, concept, rnd, gradients)

                # TODO: Allow for list of tcav scores because random samples will be larger.
                tcav_scores[bn][concept] = [test_function(rnd) for rnd in random_samples]
                
        if test:
            self.test_and_remove_concepts(tcav_scores)
        if sort:
            self._sort_concepts(tcav_scores)
        
        # Open a .pkl file in the concept directory and save the tcav scores to it.
        with open(self.cav_dir / 'tcav_scores.pkl', 'wb') as handle:
            pickle.dump(tcav_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return tcav_scores

    def do_statistical_testings(self, i_ups_concept, i_ups_random):
        """Conducts ttest to compare two set of samples.

        In particular, if the means of the two samples are staistically different.

        Args:
          i_ups_concept: samples of TCAV scores for concept vs. randoms
          i_ups_random: samples of TCAV scores for random vs. randoms

        Returns:
          p value
        """
        
        min_len = min(len(i_ups_concept), len(i_ups_random))
        _, p = stats.ttest_rel(i_ups_concept[:min_len], i_ups_random[:min_len])
        return p

    def test_and_remove_concepts(self, tcav_scores):
        """Performs statistical testing for all discovered concepts.

        Using TCAV socres of the discovered concepts versurs the random_counterpart
        concept, performs statistical testing and removes concepts that do not pass

        Args:
          tcav_scores: Calculated dicationary of tcav scores of all concepts
        """
        
        concepts_to_delete = []
        for bn in self.bottlenecks:
            for concept in self.dic[bn]['concepts']:
                pvalue = self.do_statistical_testings \
                    (tcav_scores[bn][concept], tcav_scores[bn][self.random_concept])
                if pvalue > 0.01:
                    concepts_to_delete.append((bn, concept))
        for bn, concept in concepts_to_delete:
            self.delete_concept(bn, concept)

    def delete_concept(self, bn, concept):
        """
        Removes a discovered concepts if it's not already removed.
        
        :param bn: Bottleneck layer where the concepts is discovered.
        :param concept: Concept name.
        :return: 
        """
        
        self.dic[bn].pop(concept, None)
        if concept in self.dic[bn]['concepts']:
            self.dic[bn]['concepts'].pop(self.dic[bn]['concepts'].index(concept))

    def _concept_profile(self, bn, activations, concept, randoms):
        """Transforms data points from activations space to concept space.

        Calculates concept profile of data points in the desired bottleneck
        layer's activation space for one of the concepts

        Args:
          bn: Bottleneck layer
          activations: activations of the data points in the bottleneck layer
          concept: concept name
          randoms: random concepts

        Returns:
          The projection of activations of all images on all CAV directions of
            the given concept
        """

        def t_func(rnd):
            products = self.load_cav_direction(concept, rnd, bn) * activations
            return np.sum(products, -1)

        if self.num_workers:
            pool = multiprocessing.Pool(self.num_workers)
            profiles = pool.map(lambda rnd: t_func(rnd), randoms)
        else:
            profiles = [t_func(rnd) for rnd in randoms]
        return np.stack(profiles, axis=-1)

    def find_profile(self, bn, images, mean=True):
        """Transforms images from pixel space to concept space.

        Args:
          bn: Bottleneck layer
          images: Data points to be transformed
          mean: If true, the profile of each concept would be the average inner
            product of all that concepts' CAV vectors rather than the stacked up
            version.

        Returns:
          The concept profile of input images in the bn layer.
        """
        
        profile = np.zeros((len(images), len(self.dic[bn]['concepts']),
                            self.num_random_exp))
        class_acts = get_acts_from_images(
            images, self.model, bn).reshape([len(images), -1])
        randoms = ['random500_{}'.format(i) for i in range(self.num_random_exp)]
        for i, concept in enumerate(self.dic[bn]['concepts']):
            profile[:, i, :] = self._concept_profile(bn, class_acts, concept, randoms)
        if mean:
            profile = np.mean(profile, -1)
        return profile

    def save_ace_report(self, accs=None, scores=None):
        """Saves TCAV scores.

        Saves the average CAV accuracies and average TCAV scores of the concepts
        discovered in ConceptDiscovery instance.

        """

        # If we don't have the concept dictionary, load it in.
        if not hasattr(self, "dic"):
            self.load_concept_dict()

        report_path = self.output_dir / "report.txt"

        if accs == None or scores == None:
            
            # Open the concept dictionary file and save it to self.dic
            with open(self.cav_dir / 'concept_accuracies.pkl', 'rb') as handle:
                accs = pickle.load(handle)
            
            # Open the concept dictionary file and save it to self.dic
            with open(self.cav_dir / 'tcav_scores.pkl', 'rb') as handle:
                scores = pickle.load(handle)

        report = '\n\n\t\t\t ---CAV accuracies---'


        for bn in self.bottlenecks:
            report += '\n'
            for concept in self.dic[bn]['concepts'] + [self.random_concept]:
                report += '\n' + bn + ':' + concept + ' Average accuracy: ' + str(
                    np.mean(accs[bn][concept]))

        with open(report_path, 'w') as f:
            f.write(report)

        report = '\n\n\t\t\t ---TCAV scores---'

        for bn in self.bottlenecks:
            report += '\n'
            for concept in self.dic[bn]['concepts'] + [self.random_concept]:
                pvalue = self.do_statistical_testings(
                    scores[bn][concept], scores[bn][self.random_concept])
                report += '\n{}:{}: Average Score:{} P-value:{}'.format(bn, concept,
                                                 np.mean(scores[bn][concept]), pvalue)

        with open(report_path, 'a') as f:
            f.write(report)

    def plot_concepts(self, bn, num=10, mode='diverse', concepts=None):
        """Plots examples of discovered concepts.

        Args:
        cd: The concept discovery instance
        bn: Bottleneck layer name
        num: Number of images to print out of each concept
        address: If not None, saves the output to the address as a .PNG image
        mode: If 'diverse', it prints one example of each of the target class images
          is coming from. If 'radnom', randomly samples exmples of the concept. If
          'max', prints out the most activating examples of that concept.
        concepts: If None, prints out examples of all discovered concepts.
          Otherwise, it should be either a list of concepts to print out examples of
          or just one concept's name

        Raises:
        ValueError: If the mode is invalid.
        """
        
        # If we don't have the concept dictionary, load it in.
        if not hasattr(self, "dic"):
            self.load_concept_dict()
        
        if not hasattr(self, "discovery_images"):

            # Get the list of discovery image paths.
            concept_dir = self.source_dir / self.target_class / "discovery"
            
            # Save the list of discovery images paths
            self.discovery_images = list(concept_dir.iterdir())
        
        if concepts is None:
            concepts = self.dic[bn]['concepts'] + [self.random_concept]
            
        elif not isinstance(concepts, list) and not isinstance(concepts, tuple):
            concepts = [concepts]
            
        num_concepts = len(concepts)
        plt.rcParams['figure.figsize'] = num * 2.1, 4.3 * num_concepts
        
        fig = plt.figure(figsize=(num * 2, 4 * num_concepts))
        outer = gridspec.GridSpec(num_concepts, 1, wspace=0., hspace=0.3)
        
        for n, concept in enumerate(concepts):
                
            inner = gridspec.GridSpecFromSubplotSpec(
                2, num, subplot_spec=outer[n], wspace=0, hspace=0.1)
            
            if concept == self.random_concept:
                concept_images = list((self.discovered_concepts_dir / "Random" / self.random_concept / "superpixels").iterdir())
                concept_patches = list((self.discovered_concepts_dir / "Random" / self.random_concept / "patches").iterdir())
                concept_image_numbers = [img.name for img in concept_images]
            else:
                concept_images = self.dic[bn][concept]['images']
                concept_patches = self.dic[bn][concept]['patches']
                concept_image_numbers = self.dic[bn][concept]['image_numbers']
                
            if mode == 'max':
                idxs = np.arange(len(concept_images))
            elif mode == 'random':
                idxs = np.random.permutation(np.arange(len(concept_images)))
            elif mode == 'diverse':
                idxs = []
                while True:
                    seen = set()
                    for idx in range(len(concept_images)):
                        discovery_image_num = int(concept_image_numbers[idx].split("_")[0])
                        if discovery_image_num not in seen and idx not in idxs:
                            seen.add(discovery_image_num)
                            idxs.append(idx)
                    if len(idxs) == len(concept_images):
                        break
            else:
                raise ValueError('Invalid mode!')
            idxs = idxs[:num]
            for i, idx in enumerate(idxs):
                ax = plt.Subplot(fig, inner[i])
                img = Image.open(concept_images[idx])
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == int(num / 2):
                    ax.set_title(concept)
                ax.grid(False)
                fig.add_subplot(ax)
                ax = plt.Subplot(fig, inner[i + num])
                
                concept_patch = np.array(Image.open(concept_patches[idx]))
                
                mask = 1 - (np.mean(concept_patch == float(
                    self.average_image_value) / 255, -1) == 1)
                discovery_image_num = int(concept_image_numbers[idx].split("_")[0])
                image = load_image_from_file(self.discovery_images[discovery_image_num], self.resize_dims)
                ax.imshow(mark_boundaries(image, mask, color=(1, 1, 0), mode='thick'))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(str(concept_image_numbers[idx]))
                ax.grid(False)
                fig.add_subplot(ax)
        plt.suptitle(bn)
        
    
        with open(self.output_dir / (bn + '_concepts.png'), 'wb') as f:
            fig.savefig(f)
        plt.clf()
        plt.close(fig)
