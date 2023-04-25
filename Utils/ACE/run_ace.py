from pathlib import Path
import numpy as np
import os
import sys

# This allows the other ACE files to be imported as they cannot be found in sys.path.
sys.path.append(str(Path.cwd().absolute()))

import Utils.ACE.ace_helpers as ace_helpers
from Utils.ACE.ace import ConceptDiscovery

def main():

    # Specify the target class and the source directory.
    # Note: The target class should be a folder in the source directory.
    # This folder should have a folder called discovery for images for concept discovery and a folder called tcav for tcav score calculation.
    target_class = "mitotic figure"
    source_dir = Path("D:/DS/DS4/Project/Mitotic_figures")
    
    # Create a list of the bottleneck layers.
    bottleneck_layers = ['backbone.body.layer1.2.conv1', 'backbone.body.layer2.3.conv1', 'backbone.body.layer3.5.conv1', 'backbone.body.layer4.2.conv1']

    # Create the model variable and set it to evaluate.
    mymodel = ace_helpers.MyModel("mitotic", bottleneck_layers)
    
    # List the different datasets/tissues we want to use. 
    tissue_types = ['canine cutaneous mast cell tumor', 'canine lung cancer', 'canine lymphoma', 'human breast cancer', 'human neuroendocrine tumor']
    
    # Loop through these to find potential concepts for each.
    for tissue_type in tissue_types:
        try:
        
            print(tissue_type, "Running")
            
            # create a path to the data we are currently using.
            curr_source_dir = source_dir / tissue_type

            # Create an output directory for our data.
            output = Path("D:/FYP") / f"ACE_mitotis_{tissue_type}/"
            
            # Create the directories that we intend to output files to.
            ace_helpers.create_directories(output, remove_old=False)

            # Creating the ConceptDiscovery class instance.
            cd = ConceptDiscovery(
                mymodel,
                target_class,
                curr_source_dir,
                output,
                bottleneck_layers,
                num_random_exp=25,
                channel_mean=[True, True, True, False],
                min_imgs=50,
                resize_dims=(512,512),
                pca_n_components=[600, 200, 100, 600])

            # Creating the dataset of image patches by segmenting the discovery images.
            patches_created = cd.create_patches(param_dict={'n_segments': [10]})
            
            # Once this is complete (patches_created is a boolean stating if the process ran) we will create patches from the context images.
            # The motivation for this is that we will get segments that relate to the context around the annotation, which can be used to
            # classify mitotic figures. This context may capture how close neighbouring cells are, which can sometimes be used to identify
            #mitotic figures.
            if patches_created:
                
                # Get the list of context discovery images from the directory.
                context_discovery_images = list((curr_source_dir / target_class / "context_discovery").iterdir())
                
                # Create the context patches.
                cd.create_patches(param_dict={'n_segments': [4]}, discovery_images=context_discovery_images)
                
                # We want to save the context images to our output. This will be useful later when we plot the outputs as the context
                # images contains the annotation and some additional context.
                ace_helpers.save_discovery_images(cd,save_context=True)
            
            # The following section was used as a means to update the PCA once it was already computed.
            
            # Load the superpixel images that will be used to update the PCA.
            superpixels_dir = cd.discovered_concepts_dir / "superpixels"
            superpixel_images = np.array(list(superpixels_dir.iterdir()))

            # Define the batch size that the activations were computed in (they were stored and used repeatedly as I was
            # troubleshooting memory issues).
            bs = 2
            
            # For every bottleneck and incremental PCA object.
            for bn, pca_n_components in zip(cd.bottlenecks, cd.pca_n_components):
                
                # Logic to ensure that the correct PCA object is updated.
                if bn in ['backbone.body.layer2.3.conv1', 'backbone.body.layer3.5.conv1'] and tissue_type not in ['canine cutaneous mast cell tumor']:
                    cd.update_pca(bn, pca_n_components, superpixel_images, bs)

            # If we have a PCA object, print the variance that we have preserved.
            if hasattr(cd, "pca") and cd.pca is not None:
                check_pca_variance_preserved(cd)

            print("Starting concept discovery")

            # Discovering Concepts through clustering of segment activations.
            concepts_ran = cd.discover_concepts(method='KM', param_dicts={'n_clusters': 50})
            
            # If we successfully discovered concepts, save them to the output.
            if concepts_ran:
                print("Saving concepts")
                # Save discovered concept images (resized and original sized)
                ace_helpers.save_concepts(cd)

            # Randomly select image segments to create a random concept and random groups to train the concepts against.
            # The random concept will allow us to test if our discovered concepts are statistically different from a random sample.
            # The random groups (25) will allow us to generate 25 different vectors for each concept against the random group. 
            print("Initializing random concepts")
            cd.initialize_random_concept_and_samples()

            # Now we calculate the CAVs by creating a linear classifier that splits the concept activations from the random activations.
            # The vector orthognal to the decision boundary gives us the vector that represents the concept.
            print("Computing CAVs")
            cav_accuracies = cd.cavs()

            # Now we compute the TCAV scores by getting the product of our gradients by the CAV and assessing how many of the predictions
            # it had a positive impact on.
            print("Computing TCAV scores")
            scores = cd.tcavs(test=False, sort=False)

            # Generate a textual report that contains how well the CAVs were split from the random, the average TCAV score of the concepts
            # and a p-value for a 2 sided t-test to see if the concepts tcav scores were statistically different from the random concept's scores.
            print("Generating report")
            cd.save_ace_report()

            # Plot examples of discovered concepts and also the concepts that were influential and statistically different from the random
            # concept.
            for bn in cd.bottlenecks:
                cd.plot_concepts(bn)
                cd.plot_influential_concepts(bn, 30, 10)
        
        # If we encounter an exception, print it and move on to the next tissue.
        except Exception as e:
            print(e)


def check_pca_variance_preserved(cd):
    """
    This function takes a ConceptDiscovery instance and prints the variance
    preserved in the IncrementalPCA object for each bottleneck."""
    
    # For every layer and PCA instance.
    for layer, pca_instance in cd.pca.items():
        
        # Print the cumulative variance explained.
        print(layer, pca_instance.explained_variance_ratio_.cumsum()[-1])
    
if __name__ == "__main__":
    main()
