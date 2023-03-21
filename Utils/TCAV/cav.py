
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


def flatten_activations_and_get_labels(concepts, layer_name, activations):
    '''
    :param concepts: different name of concepts
    :param layer_name: the name of the layer to compute CAV on
    :param activations: activations with the size of num_concepts * num_layers * num_samples
    :return:
    '''
    # in case of different number of samples for each concept
    min_num_samples = np.min([activations[c][layer_name].shape[0] for c in concepts])
    # flatten the activations and mark the concept label
    data = []
    concept_labels = np.zeros(len(concepts) * min_num_samples)
    label_concept = {}
    for i, c in enumerate(concepts):
        data.extend(activations[c][layer_name][:min_num_samples].reshape(min_num_samples, -1))
        concept_labels[i * min_num_samples : (i + 1) * min_num_samples] = i
        label_concept[i] = c
    data = np.array(data)
    return data, concept_labels, label_concept


class CAV(object):
    def __init__(self, concepts, layer_name, save_path, hparams=None):
        self.concepts = concepts
        self.layer_name = layer_name
        self.save_path = save_path
        
        if hparams:
            self.hparams = hparams
        else:
            self.hparams = {'model_type':'linear', 'alpha':.01}
    
    def cav_filename(self):
        concepts = "_".join([str(c) for c in self.concepts])
        model_type = self.hparams["model_type"]
        alpha = self.hparams["alpha"]
        
        return f"{concepts}_{self.layer_name}_{model_type}_{alpha}.pkl"

    def train(self, activations):
        data, labels, label_concept = flatten_activations_and_get_labels(self.concepts, self.layer_name, activations)
        # print("Data shape:", data.shape)

        # default setting is One-Vs-All
        assert self.hparams["model_type"] in ['linear', 'logistic']
        if self.hparams["model_type"] == 'linear':
            model = SGDClassifier(alpha=self.hparams["alpha"])
        else:
            model = LogisticRegression()

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, stratify=labels)
        
        model.fit(x_train, y_train)
        
        # Get accuracy
        y_pred = model.predict(x_test)
        # get acc for each class.
        num_classes = max(labels) + 1
        acc = {}
        num_correct = 0
        
        for class_id in range(int(num_classes)):
            
            # get indices of all test data that has this class.
            idx = (y_test == class_id)
            
            acc[label_concept[class_id]] = metrics.accuracy_score(y_pred[idx], y_test[idx])

              # overall correctness is weighted by the number of examples in this class.
            num_correct += (sum(idx) * acc[label_concept[class_id]])
            
        acc['overall'] = float(num_correct) / float(len(y_test))
            
        self.accuracies = acc
        
        '''
        The coef_ attribute is the coefficients in linear regression.
        Suppose y = w0 + w1x1 + w2x2 + ... + wnxn
        Then coef_ = (w0, w1, w2, ..., wn). 
        This is exactly the normal vector for the decision hyperplane
        '''
        
        
        if len(model.coef_) == 1:
            self.cavs = np.array([-model.coef_[0], model.coef_[0]])
        else:
            self.cavs = -np.array(model.coef_)
        
        self.save_cav()

    def get_cav(self, concept):
        return self.cavs[self.concepts.index(concept)]
    
    def save_cav(self):
        """Save a dictionary of this CAV to a pickle."""
        
        save_dict = {
            'concepts': self.concepts,
            'bottleneck': self.layer_name,
            'hparams': self.hparams,
            'accuracies': self.accuracies,
            'cavs': self.cavs,
            'saved_path': self.save_path
            }
        
        if self.save_path is not None:
            with open(self.save_path / self.cav_filename(), 'wb') as pkl_file:
                pickle.dump(save_dict, pkl_file)
    
    def load_cav(cav_path):
        """Make a CAV instance from a saved CAV (pickle file).
            Args:
            cav_path: the location of the saved CAV
            Returns:
            CAV instance.
        """
        
        with open(cav_path, 'rb') as pkl_file:
            save_dict = pickle.load(pkl_file)

        cav = CAV(save_dict['concepts'], save_dict['bottleneck'], 
                  save_dict['hparams'], save_dict['saved_path'])
        
        cav.accuracies = save_dict['accuracies']
        cav.cavs = save_dict['cavs']
        return cav

def load_or_train_cav(concepts, layer_name, save_path, hparams=None, activations=None, overwrite=False):
    cav_instance = CAV(concepts, layer_name, save_path, hparams)
    
    if save_path is not None:
        cav_path = Path(save_path) / cav_instance.cav_filename()
        
    if not overwrite and cav_path.is_file():
        cav_instance = CAV.load_cav(cav_path)
        return cav_instance

    if activations is not None:
        cav_instance.train(activations)
        return cav_instance
