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
    
#     tissue_types = ['canine cutaneous mast cell tumor', 'canine lung cancer', 'canine lymphoma', 'human breast cancer', 'human neuroendocrine tumor']
    tissue_types = ['canine lung cancer']
                
    for tissue_type in tissue_types:
        try:
        
            print(tissue_type, "Running")


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
                num_random_exp=25,
                channel_mean=[True, True, True, False],
                min_imgs=50,
                resize_dims=(512,512),
                pca_n_components=[600, 200, 100, 600])

            # Creating the dataset of image patches.
#             patches_created = cd.create_patches(param_dict={'n_segments': [10]})

#             if patches_created:
#                 context_discovery_images = list((curr_source_dir / target_class / "context_discovery").iterdir())
#                 # Create the context patches
#                 cd.create_patches(param_dict={'n_segments': [4]}, discovery_images=context_discovery_images)

#                 ace_helpers.save_discovery_images(cd,save_context=True)

#     #         superpixels_dir = cd.discovered_concepts_dir / "superpixels"
#     #         superpixel_images = np.array(list(superpixels_dir.iterdir()))

#     #         bs = 2

#     #         for bn, pca_n_components in zip(cd.bottlenecks, cd.pca_n_components):

#     #             if bn in ['backbone.body.layer2.3.conv1', 'backbone.body.layer3.5.conv1'] and tissue_type not in ['canine cutaneous mast cell tumor']:
#     #                 cd.update_pca(bn, pca_n_components, superpixel_images, bs)


#     #             if tissue_type == "canine lymphoma" and bn == "backbone.body.layer4.2.conv1":
#     #                 cd.update_pca(bn, pca_n_components, superpixel_images, bs)

#             if hasattr(cd, "pca") and cd.pca is not None:
#                 check_pca_variance_preserved(cd)

#             print("Starting concept discovery")

#             # Discovering Concepts
#             concepts_ran = cd.discover_concepts(method='KM', param_dicts={'n_clusters': 50})

#             if concepts_ran:
#                 print("Saving concepts")
#                 # Save discovered concept images (resized and original sized)
#                 ace_helpers.save_concepts(cd)

#             print("Initializing random concepts")
#             cd.initialize_random_concept_and_samples()

#             print("Computing CAVs")
#             cav_accuracies = cd.cavs()

#             print("Computing TCAV scores")
#             scores = cd.tcavs(test=False, sort=False)

#             print("Generating report")
#             cd.save_ace_report()

            # Plot examples of discovered concepts
            for bn in cd.bottlenecks:
                cd.plot_influential_concepts(bn, 30, 10)
                
        except Exception as e:
            print(e)


def check_pca_variance_preserved(cd):
    
    for layer, pca_instance in cd.pca.items():
        print(layer, pca_instance.explained_variance_ratio_.cumsum()[-1])
    
    
                         

if __name__ == "__main__":
    main()
