from pathlib import Path
import numpy as np

import Utils.ACE.ace_helpers as ace_helpers
from Utils.ACE.ace import ConceptDiscovery

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
