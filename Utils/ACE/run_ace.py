from pathlib import Path
import numpy as np
import os
import sys

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
    
    tissue_types = ['canine cutaneous mast cell tumor', 'canine lung cancer', 'canine lymphoma', 'human breast cancer', 'human neuroendocrine tumor']
    
    for tissue_type in tissue_types:
        
        curr_source_dir = source_dir / tissue_type
    
        # Create an output directory for our data
        output = Path("D:/FYP") / f"ACE_mitotis_{tissue_type}/"
        
        ace_helpers.create_directories(output, remove_old=False)

        # Creating the ConceptDiscovery class instance.
        cd = ConceptDiscovery(
            mymodel,
            target_class,
            curr_source_dir,
            output,
            bottleneck_layers,
            num_random_exp=50,
            channel_mean=[True, True, False, False],
            min_imgs=100,
            resize_dims=(512,512),
            pca_n_components=[250, 125, 800, 565])
        
        # Creating the dataset of image patches.
        patches_created = cd.create_patches(param_dict={'n_segments': [10]})
        
        if patches_created:
            context_discovery_images = list((curr_source_dir / target_class / "context_discovery").iterdir())
            # Create the context patches
            cd.create_patches(param_dict={'n_segments': [4]}, discovery_images=context_discovery_images)

            ace_helpers.save_discovery_images(cd,save_context=True)
        
        # Discovering Concepts
        cd.discover_concepts(method='KM', param_dicts={'n_clusters': 50})
        
        # Save discovered concept images (resized and original sized)
        ace_helpers.save_concepts(cd)
        
#         cd.initialize_random_concept_and_samples()
        
#         cav_accuracies = cd.cavs()
        
#         scores = cd.tcavs(test=False, sort=False)
        
#         cd.save_ace_report()
        
#         # Plot examples of discovered concepts
#         for bn in cd.bottlenecks:
#             cd.plot_concepts(bn, 10)


def run_concept_discovery(output_dir, target_class, source_dir, model_params, cd_params, patch_params, clustering_params, cav_params, tcav_params):
    # Create an output directory for our data
    output_dir = Path(output_dir)

    ace_helpers.create_directories(output)


    # Create the model variable and set it to evaluate.
    mymodel = ace_helpers.MyModel(**model_params)
    mymodel.model.eval()

    # Creating the ConceptDiscovery class instance.
    cd = ConceptDiscovery(
        mymodel,
        target_class,
        source_dir,
        output_dir,
        **cd_params)

    # Creating the dataset of image patches.
    cd.create_patches(**patch_params)

    # Saving the concept discovery target class images.
    image_dir = cd.discovered_concepts_dir / 'images'
    image_dir.mkdir()
    ace_helpers.save_images(image_dir.absolute(),
                            (cd.discovery_images * 256).astype(np.uint8))

    # Discovering Concepts
    cd.discover_concepts(**clustering_params)

    # Save discovered concept images (resized and original sized)
    ace_helpers.save_concepts(cd)

    cd.initialize_random_concept_and_samples()

    cav_accuracies = cd.cavs(**cav_params)

    scores = cd.tcavs(**tcav_params)

    cd.save_ace_report()

    # Plot examples of discovered concepts
#         for bn in cd.bottlenecks:
#             ace_helpers.plot_concepts(cd, bn, 10, address=results_dir)

if __name__ == "__main__":
    main()
