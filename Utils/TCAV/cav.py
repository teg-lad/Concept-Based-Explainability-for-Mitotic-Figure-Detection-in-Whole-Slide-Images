import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


def flatten_activations_and_get_labels(concepts, layer_name, activations):
    """
    This function takes a concept, layer name and activations and returns the flatten activations
    ready for use by the linear model.
    :param concepts: different name of concepts
    :param layer_name: the name of the layer to compute CAV on
    :param activations: activations with the size of num_concepts * num_layers * num_samples
    :return:
    Data, labels and the concept names.
    """

    # in case of different number of samples for each concept
    min_num_samples = np.min([activations[c][layer_name].shape[0] for c in concepts])

    # flatten the activations and mark the concept label
    data = []
    concept_labels = np.zeros(len(concepts) * min_num_samples)
    label_concept = {}

    # For every concept (concept and random concept), add samples to the list and add labels.
    for i, c in enumerate(concepts):
        data.extend(activations[c][layer_name][:min_num_samples].reshape(min_num_samples, -1))
        concept_labels[i * min_num_samples: (i + 1) * min_num_samples] = i
        label_concept[i] = c

    # Create an array from the data and return.
    data = np.array(data)

    return data, concept_labels, label_concept


class CAV(object):
    """
    This class implements a CAV object that takes a concept and negative/random concept.
    It can be trained by passing activations for these 2 concepts and a vector can be extracted
    that corresponds to the vector orthogonal to the decision boundary.
    """

    def __init__(self, concepts, layer_name, save_path, hparams=None):

        self.concepts = concepts
        self.layer_name = layer_name
        self.save_path = save_path

        # If hyperparameters were supplied use them, else use the default.
        if hparams:
            self.hparams = hparams
        else:
            self.hparams = {'model_type': 'linear', 'alpha': .01}

    def cav_filename(self):
        """
        This function returns the filename that will be given to this CAV based on the concepts,
        bottleneck and linear model used.
        """

        # Create a string of the concepts.
        concepts = "_".join([str(c) for c in self.concepts])

        # Get the linear model type and alpha.
        model_type = self.hparams["model_type"]
        alpha = self.hparams["alpha"]

        # Return the filename.
        return f"{concepts}_{self.layer_name}_{model_type}_{alpha}.pkl"

    def train(self, activations):
        """
        This function takes the activations for the concept and random concept images and uses
        them to create a linear classifier that separates the two groups. The vector that is orthogonal
        to this hyperplane that is the decision boundary is the values of the CAV.
        :param activations: A dictionary containing the activations for each concept with the layers as keys.
        """

        # Flatten and return the activations.
        data, labels, label_concept = flatten_activations_and_get_labels(self.concepts, self.layer_name, activations)

        # default setting is One-Vs-All
        assert self.hparams["model_type"] in ['linear', 'logistic']
        if self.hparams["model_type"] == 'linear':
            model = SGDClassifier(alpha=self.hparams["alpha"])
        else:
            model = LogisticRegression()

        # Split the data into train and test for use with the model.
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, stratify=labels)

        # For the model on the train data.
        model.fit(x_train, y_train)

        # Get the predictions
        y_pred = model.predict(x_test)

        # get acc for each class.
        num_classes = max(labels) + 1
        acc = {}
        num_correct = 0

        for class_id in range(int(num_classes)):
            # get indices of all test data that has this class.
            idx = (y_test == class_id)

            # get the accuracy for this class.
            acc[label_concept[class_id]] = metrics.accuracy_score(y_pred[idx], y_test[idx])

            # overall correctness is weighted by the number of examples in this class.
            num_correct += (sum(idx) * acc[label_concept[class_id]])

        # Accuracy is the number of correct over the total samples.
        acc['overall'] = float(num_correct) / float(len(y_test))

        # Add the accuracy to the object.
        self.accuracies = acc

        # The coef_ attribute is the coefficients in linear regression.
        # Suppose y = w0 + w1x1 + w2x2 + ... + wnxn
        # Then coef_ = (w0, w1, w2, ..., wn). 
        # This is exactly the normal vector for the decision hyperplane

        # Extract the coefficients.
        if len(model.coef_) == 1:
            self.cavs = np.array([-model.coef_[0], model.coef_[0]])
        else:
            self.cavs = -np.array(model.coef_)

        # Save the CAV for use later.
        self.save_cav()

    def get_cav(self, concept):
        """
        This function gets the CAV vector from the object.
        """

        return self.cavs[self.concepts.index(concept)]

    def save_cav(self):
        """Save a dictionary of this CAV to a pickle."""

        # Create the dictionary that we save to store the attributes of this CAV.
        save_dict = {
            'concepts': self.concepts,
            'bottleneck': self.layer_name,
            'hparams': self.hparams,
            'accuracies': self.accuracies,
            'cavs': self.cavs,
            'saved_path': self.save_path
        }

        # If a save path exists, write the dictionary to it.
        if self.save_path is not None:
            with open(self.save_path / self.cav_filename(), 'wb') as pkl_file:
                pickle.dump(save_dict, pkl_file)


def load_cav(cav_path):
    """
    Make a CAV instance from a saved CAV (pickle file).
    :param cav_path: the location of the saved CAV.
    :return: CAV instance.
    """

    # Open the path and read in the pickled dictionary.
    with open(cav_path, 'rb') as pkl_file:
        save_dict = pickle.load(pkl_file)

    # Pass the values from the dictionary into a new CAV object.
    cav = CAV(save_dict['concepts'], save_dict['bottleneck'],
              save_dict['hparams'], save_dict['saved_path'])

    # Pass the values to the CAV attributes.
    cav.accuracies = save_dict['accuracies']
    cav.cavs = save_dict['cavs']

    # Return this loaded CAV instance.
    return cav


def load_or_train_cav(concepts, layer_name, save_path, hparams=None, activations=None, overwrite=False):
    """
    This function takes information about the concept, bottleneck layer and the save path and either loads a CAV if it
    exists in the save path directory or creates a new instance that is ready to be trained.
    :param concepts:
    :param layer_name:
    :param save_path:
    :param hparams:
    :param activations:
    :param overwrite:
    :return:
    """

    # Create a CAV instance with the supplied arguments.
    cav_instance = CAV(concepts, layer_name, save_path, hparams)

    # If a save_path is defined, create the path to the CAV if it were to exist.
    if save_path is not None:
        cav_path = Path(save_path) / cav_instance.cav_filename()

    # If we aren't overwriting an existing CAV and the file exists.
    if not overwrite and cav_path.is_file():

        # Open and load the CAV.
        cav_instance = load_cav(cav_path)

        # Return the CAV instance.
        return cav_instance

    # If activations were passed.
    if activations is not None:

        # Train the CAV with the activations.
        cav_instance.train(activations)

        # Return the CAV instance.
        return cav_instance
