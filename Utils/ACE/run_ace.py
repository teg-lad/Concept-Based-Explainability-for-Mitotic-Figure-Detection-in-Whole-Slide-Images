from pathlib import Path
import numpy as np

import Utils.ACE.ace_helpers as ace_helpers
from Utils.ACE.ace import ConceptDiscovery

def main():
    # Create an output directory for our data
    output = Path.cwd() / "ACE_mitotis_test/"

    # Create the sub directories at the output location.
    ace_helpers.create_directories(output, remove_old=False)
    
    # Specify the target class and the source directory.
    # Note: The target class should be a folder in the source directory.
    # This folder should have a folder called discovery for images for concept discovery and a folder called tcav for tcav score calculation.
    target_class = "mitotic figure"
    source_dir = "D:/DS/DS4/Project/Mitotic_figures/canine lymphoma"
    
    # Create a list of the bottleneck layers.
    bottleneck_layers = ['backbone.body.layer1.2.conv1', 'backbone.body.layer2.3.conv1', 'backbone.body.layer3.5.conv1', 'backbone.body.layer4.2.conv1']

    # Create the model variable and set it to evaluate.
    mymodel = ace_helpers.MyModel("tmp", bottleneck_layers)
    mymodel.model.eval()
    mymodel.model.model
    
    # Creating the ConceptDiscovery class instance.
    cd = ConceptDiscovery(
        mymodel,
        target_class,
        source_dir,
        output,
        bottleneck_layers,
        num_random_exp=2,
        channel_mean=False,
        min_imgs=50,
        resize_dims=(512,512), pca_n_components=100)
    
    # Get the superpixel images.
    superpixels_dir = cd.discovered_concepts_dir / "superpixels"
    superpixel_images = np.array(list(superpixels_dir.iterdir()))
    
    # Get the activations back after passing the superpixels.
    activations = cd._get_activations(superpixel_images, bs=2)
    
    print([acts.shape for key, acts in activations.items()])


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
