"""
This file contains functions for carrying out the training for this project.
These functions make use of the PyTorch Object Detection Finetuning Tutorial and other related PyTorch resources.
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

from pathlib import Path
from tqdm import tqdm
import math
from datetime import datetime
import pickle

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import Utils.Training.utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, writer, scaler=None):
    """
    This method takes a model, optimizer and dataloader and carries out one epoch of training.
    This involves logging metrics, calculating the training loss, computing the gradients and updating the weights.
    The training loss is written
    :param model: A PyTorch Faster-RCNN model.
    :param optimizer: A PyTorch optimizer
    :param data_loader: A PyTorch DataLoader object.
    :param device: The device where training should take place.
    :param epoch: The current epoch number, for output purposes.
    :param print_freq: Number of batches to process before logging output is printed to stdout.
    :param writer: The TensorBoard SummaryWriter object to write to.
    :param scaler: Gradient scaler for use when autocasting for mixed precision.
    :return: Metric logger and the training loss.
    """

    # Set the model to training model
    model.train()

    # Set up the metric logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # If we are at our first epoch we can warm up using a learning rate scheduler.
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    # Variable for storing the cumulative loss.
    total_loss = 0

    # Iterate through the training data loader
    for iteration, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Pull the images and target from the data tuple
        images, targets = data

        # Move the data to the device the computation should happen on
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k in ["boxes", "labels"]} for t in targets]

        # With autocasting if we have a scaler defined.
        with torch.cuda.amp.autocast(enabled=scaler is not None):

            # Get the losses, and sum for the total loss.
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes.
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        # Add the loss from this batch to the total.
        total_loss += loss_value

        # Delete the images, targets and loss_dict to free up memory on the device.
        del images, targets, loss_dict

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        # Zero the gradient and compute the gradients and take a step with the optimizer.
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update the metric logger
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Compute the average loss per batch
    train_loss = total_loss / len(data_loader)

    # Write the training loss to the TensorBoard SummaryWriter
    writer.add_scalar('training loss', train_loss, epoch)

    return metric_logger, train_loss


def evaluate(model, data_loader_test, epoch, device, writer):
    """
    This method takes the model, dataloader and evaluates the model on the test data in the dataloader.
    :param model: A PyTorch Faster-RCNN model.
    :param data_loader_test: A PyTorch DataLoader object with the test data.
    :param epoch: The epoch number for logging purposes.
    :param device: The device where evaluation should take place.
    :param writer: The TensorBoard SummaryWriter object to write to.
    :return: Test loss and the mAP metrics from TorchMetrics.
    """
    # Set the model to evaluation mode
    model.eval()

    # initialize and instance of the TorchMetric MeanAveragePrecision class.
    metric = MeanAveragePrecision()
    metric.to(device)

    # Variable for storing the loss
    total_loss = 0

    # With no gradient computation to save memory as we don't need the gradient as we won't update the weights.
    with torch.no_grad():
        # Iterate through the images
        for images, targets in tqdm(data_loader_test):
            # Move the data to the device
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items() if k in ["boxes", "labels"]} for t in targets]

            # Get the loss and predictions back from the model.
            losses, predictions, _ = model(images, targets)

            # Sum the losses to get the loss for this batch and add it to the total.
            losses_summed = sum(loss for loss in losses.values()).item()
            total_loss += losses_summed

            # Add the predictions and ground truths from this batch to the MeanAveragePrecision metric.
            metric.update(predictions, targets)

            # Delete the predictions, images, targets and losses to free up memory on the device.
            del predictions, images, targets, losses

    # Get the average test loss per batch by dividing the total loss by the number of batches.
    test_loss = total_loss / len(data_loader_test)

    # Write the test loss to the TensorBoard SummaryWriter.
    writer.add_scalar('test loss', test_loss, epoch)

    # Empty the cache to make memory available for the next training iteration.
    torch.cuda.empty_cache()

    # Compute the MeanAveragePrecision metric with the given predictions and ground truths.
    test_metrics = metric.compute()

    return test_loss, test_metrics


def train(model, dataset, num_epochs, output_dir, writer, checkpoint=None):
    """
    This method takes a model, number of epochs to train and the checkpoint if the model has alreadu been trained for
    several epochs.
    :param model: A PyTorch Faster-RCNN model.
    :param num_epochs: The number of epochs to train for.
    :param output_dir: Directory in which to save models and metrics.
    :param writer: The TensorBoard SummaryWriter object to write to.
    :param checkpoint: A checkpoint that contains a state_dict for the model, optimizer and the current epoch.
    """

    # Train on the GPU or on the CPU, if a GPU is not available.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Split the dataset in train and test set.
    # Note: Using torch.manual_seed(42) ensures we get the same train and test split for each run of this method.
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:-750])
    dataset_test = torch.utils.data.Subset(dataset, indices[-750:])

    torch.manual_seed(42)
    # Define training and test data loaders.
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=False, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, collate_fn=utils.collate_fn)

    # If we have a checkpoint, we need to load the state dict into the model so we get the current weights.
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Move the model to the right device.
    model.to(device)

    # Construct an optimizer.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=0.0005)

    epoch = 0

    # If we have a checkpoint, load the state dict of the optimizer into the current optimizer.
    # Also load the epoch and then delete the checkpoint object as we are finished with it.
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        del checkpoint

    # Set up a learning rate scheduler, this does not have to be saved as we pass the last epoch after initializing.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    lr_scheduler.last_epoch = epoch

    # For each epoch from the current until we have carried out as many as specified.
    for epoch in range(epoch, epoch + num_epochs):
        # Train for one epoch, printing every 100 iterations.
        _, training_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, 100, writer)

        # Update the learning rate.
        lr_scheduler.step()

        # Evaluate on the test dataset.
        test_loss, test_metrics = evaluate(model, data_loader_test, epoch, device, writer)

        # Create/open the metrics file and print the loss for both training and test and also the mAP metrics.
        with open(Path(output_dir) / "metrics.txt", "a") as m:
            m.write(f"Epoch {epoch}: " + f"Training loss: {training_loss} Test loss: {test_loss}\n" + str(
                test_metrics) + "\n\n")

        # Get the current timestamp for saving the model.
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Create the path to save the model to.
        path_to_save = Path(output_dir) / f"{timestamp}_{epoch}.pth"

        # Define what variables are to be saved to the file.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path_to_save)


def train_ae(model, dataset=None, transform=None, inv_transform=None, num_epochs=10, bs=2, lr=0.5, momentum=0.,
             output_path=None, checkpoint=None, **kwargs):

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    loss = None  # just to avoid reference before assigment
    history = {'tr_loss': [], 'val_loss': []}

    if checkpoint:
        checkpoint_path = Path(checkpoint)
        metrics_path = Path(checkpoint.rsplit(".", 1)[0] + "_metrics.pkl")

        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        # Open the activation file and read it in.
        with open(metrics_path, 'rb') as handle:
            history = pickle.load(handle)

    # Train on the GPU or on the CPU, if a GPU is not available.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Move the model to the device
    model.to(device)

    # Split the dataset in train and test set.
    # Note: Using torch.manual_seed(42) ensures we get the same train and test split for each run of this method.
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:-750])
    dataset_test = torch.utils.data.Subset(dataset, indices[-750:])

    torch.manual_seed(42)
    # Define training and test data loaders.
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=bs, shuffle=False, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=bs, shuffle=False, collate_fn=utils.collate_fn)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = torch.nn.MSELoss()

    current_epoch = 0

    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch'] + 1
        del checkpoint

    torch.cuda.empty_cache()

    # Resize
    # resize_transform = T.Resize((400, 400))

    for epoch in range(current_epoch, current_epoch + num_epochs):
        # training
        model.train()
        tr_loss = 0

        # Iterate through the training data loader
        for iteration, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            # zero the gradient
            optimizer.zero_grad()

            # Pull the images and target from the data tuple
            images, _ = data
            images = images.to(device)

            # Transform the images.
            transformed_images, _ = transform(images, _)
            transformed_images = transformed_images.tensors

            # Move the data to the device the computation should happen on
            # transformed_images = transformed_images.to(device)

            # Calculate the outputs.
            outputs = model(transformed_images)

            # transformed_images = resize_transform(transformed_images)
            inverse_transformed_output = inv_transform(outputs)

            # compute loss (flatten output in case of ConvAE. targets already flat)
            loss = criterion(inverse_transformed_output, images)
            tr_loss += loss.item()

            # propagate back the loss
            loss.backward()
            optimizer.step()

        last_batch_loss = loss.item()
        tr_loss /= len(data_loader)
        history['tr_loss'].append(round(tr_loss, 5))

        # validation
        val_loss = evaluate_ae(model, criterion, data_loader=data_loader_test, transform=transform, inv_transform=inv_transform, device=device)
        history['val_loss'].append(round(val_loss, 5))
        torch.cuda.empty_cache()

        # Get the current timestamp for saving the model.
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Create the path to save the model to.
        path_to_save = output_path / f"ResNetAE_{timestamp}_{epoch}.pth"

        # Define what variables are to be saved to the file.
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path_to_save)

        metrics_path = output_path / f"ResNetAE_{timestamp}_{epoch}_metrics.pkl"

        with open(metrics_path, 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # simple early stopping mechanism
        # if epoch >= 10:
        #     last_values = history['val_loss'][-10:]
        #     if last_values[-3] < last_values[-2] < last_values[-1]:
        #         return history

    return history


def evaluate_ae(model, criterion, data_loader=None, transform=None, inv_transform=None, device=None, **kwargs):
    """ Evaluate the model """

    # evaluate
    model.to(device)
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for iteration, data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating Model"):
            # Pull the images and target from the data tuple
            images, _ = data
            images = images.to(device)

            # Transform the images.
            transformed_images, _ = transform(images, _)
            transformed_images = transformed_images.tensors

            # transformed_images = transformed_images.to(device)

            # Calculate the outputs.
            outputs = model(transformed_images)

            inv_transformed_output = inv_transform(outputs)

            # flatten outputs in case of ConvAE (targets already flat)
            loss = criterion(inv_transformed_output, images)
            val_loss += loss.item()

    return val_loss / len(data_loader)
