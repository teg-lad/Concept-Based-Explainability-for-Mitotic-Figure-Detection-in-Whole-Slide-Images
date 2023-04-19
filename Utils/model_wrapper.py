from torch.autograd import grad
import torch
from torchvision import transforms


class ModelWrapper(object):
    """
    This class takes a model and is used to implement hooks in both the forward and backwards pass to catch the activations
    and gradients respectively. The gradient implementation is carried out for multiple sets of class logits so the 
    gradients for many different detections in one image can be obtained."""
    
    def __init__(self, model, layers):
        """
        This init function takes the model and the layer that we want to pull the activations and gradients from.
        It carries out the insertion of the forward hooks into the needed layers, the backward hooks are inserted
        when the generate_gradients method is called.
        
        model: The model to be used.
        layers: The layers in which hooks should be placed.
        """
        # Save the model and layer values in instance variables
        self.model = model
        self.layers = layers
        
        # Create instance variables to store the activations and the gradients.
        self.intermediate_activations = {}
        self.gradients = {}

        
        def save_activation(name):
            """
            This function takes a name and returns a hook that can be used in a register_forward_hook method.
            
            name: The name for the activation to be saved under in the intermediate_activations dict.
            """

            # Define the hook, the input parameters if this hook is to be passed to the register_forward_hook method.
            def hook(module, input, output):
                
                # Save the detached activation tensor to the name given in the intermediate_activations dict.
                self.intermediate_activations[name] = output

            return hook
        
        # Iterate through all of the model parameters.
        for name, param in model.named_parameters():
            
            # If the name, ignoring the .weight element at the end, is present in the list of layers we need
            # to extract the gradients for, set requires_grad to True.
            if name.rsplit(".", 1)[0] in layers:
                param.requires_grad = True
        
        # For every layer, we need to register a hook.
        for bn in layers:
    
            # We need to break the layer string into the constituent parts.
            # Note: This part is highly model dependent, you may need to alter the accessing of attributes to get to the tensor
            # you need to extract.
            module_attribute, body_attribute, layer_attribute, layer_number_attribute, component_atribute = bn.split(".")
            
            # Get the attribute for the module.
            module = getattr(self.model, module_attribute)
            
            # Get the attribute for the body.
            body = getattr(module, body_attribute)
            
            # Get the attribute for the layer.
            layer = getattr(body, layer_attribute)
            
            # Get the entry in the layer.
            layer_number = layer[int(layer_number_attribute)]
            
            # Get the specific component within that part of the layer.
            component = getattr(layer_number, component_atribute)
            
            # Register a hook in that component to save the activations that it outputs.
            component.register_forward_hook(save_activation(bn))
            
    

    def save_gradient(self, name):
        """
        This function takes a name and creates a hook to be used to register a hook to save the gradients.
        
        name: The name for the gradient to be saved under in the gradients dict.
        """
        
        # Define a hook to be used in register hook.
        def hook(grad):
            
            # Save the detached gradient tensor to the name given in the gradients dict.
            self.gradients[name] = grad.cpu().detach()

        return hook

    def generate_gradients(self, c):
        """
        This function inserts hooks into activation tensors so that the gradients can be extracted when a backwards pass of
        the network is carried out. This function works best with one image being ran through at a time, though it will work
        with multiple. The gradients will just need to be split after they are returned as they will be calculated wrt all
        images in the input as opposed to just the image the detection resides in."""
        
        # Initialize storage for the gradients and the detection info.
        grad_output = {}
        detection_info = []
        gradients_found = False
        
        # For every layer.
        for bn in self.layers:
            
            # retreive the activation tensor
            activation = self.intermediate_activations[bn]
            
            # Insert a hook to extract the gradients on the back pass.
            activation.register_hook(self.save_gradient(bn))
            
            # Create a list under the layer name so we can store all of the gradients for all of the detections.
            grad_output[bn] = []
        
        # For every set of predictions and logits (They are returned in batches).
        for preds, logits in zip(self.predictions, self.class_logits):
            
            # Extract the boxes from the predictions
            boxes = preds["boxes"]
            
            # For detection logit.
            for i in range(logits.size()[0]):
                
                # Get the corresponding box and add it to the detection info.
                box = boxes[i]
                detection_info.append("_".join([str(int(v)) for v in box.cpu().detach().squeeze().tolist()]))
                
                # Get this specific logit and call backward
                logit = logits[i, c]
                logit.backward(torch.ones_like(logit), retain_graph=True)
                gradients_found = True
                
                # For every layer we have hooks in, save their gradient.
                for bn in self.layers:
                    grad_output[bn].append(self.gradients[bn].cpu().detach())
                    
        # Check that we have gradients and that there has been a predictions
        if gradients_found:
            # For every layer we can concatenate the tensors into one tensor.
            for bn in self.layers:
                grad_output[bn] = torch.cat(grad_output[bn], dim=0)
                
        # Delete these to save memory.
        del self.predictions
        del self.class_logits
    
        return grad_output, detection_info

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def __call__(self, x):
        self.predictions, self.class_logits = self.model(x)
        return self.predictions, self.class_logits
