"""
End-to-End Adversarial Erasing for Weakly Supervised Semantic Segmentation (EADER)
"""
import argparse
import os

import torch

from data.load_data import load_data
from data.pascal_voc import LABELS, NUM_CLASSES # TODO: improve this
from models.load_models import load_localizer, load_adversarial
from utils import save_gradcam, tensor2imwrite, gcam_to_mask, denormalize, generate_segmentation_map,\
                  save_segmentation_map, erase_mask
from training_utils import TrainingPhase, compute_am_loss, load_optimizer, compute_regularization_loss
from metrics import Metrics, AverageMeter

PARSER = argparse.ArgumentParser(
                    description='End-to-End Adversarial Erasing for Weakly Supervised Semantic Segmentation')
PARSER.add_argument('--experiment_name', default='test', type=str,
                    help='Name of the current experiment, which is used to name the results folder')
PARSER.add_argument('--dataset_root', default='data', type=str, help='Root directory of the dataset')
PARSER.add_argument('--download_dataset', action='store_true', help='Download the dataset')
PARSER.add_argument('--localizer_model', default='resnet101', type=str, help='Model to use as feature extractor')
PARSER.add_argument('--localizer_from_scratch', action='store_true',
                    help='Whether to train the localizer from scratch (instead of using ImageNet weights')
PARSER.add_argument('--adversarial_model', default=None, type=str,
                    help='Model to use when classifying the remaining image after erasing the mask')
PARSER.add_argument('--adversarial_from_scratch', action='store_true',
                    help='Whether to train the adversarial from scratch (instead of using ImageNet weights')
PARSER.add_argument('--epochs', default=10, type=int, help='Number of epoch to train the model for')
PARSER.add_argument('--batch_size', default=16, type=int, help='Batch size to use during training and testing')
PARSER.add_argument('--img_resolution', default=448, type=int,
                    help='Image resolution (both height and width) to use as input')
PARSER.add_argument('--learning_rate', default=0.01, type=float, help='Learning rate to use for the localizer')
PARSER.add_argument('--adversarial_learning_rate', default=0.01, type=float,
                    help='Learning rate to use for the adversarial model')
PARSER.add_argument('--optimizer', default='sgd', type=str, help='Optimizer to use for the localizer')
PARSER.add_argument('--adversarial_optimizer', default='sgd', type=str,
                    help='Optimizer to use for the adversarial model')
PARSER.add_argument('--resume', default='', help='Path to checkpoint to resume from')
PARSER.add_argument('--evaluate', action='store_true', help='Only evaluate the model, given a model checkpoint')
PARSER.add_argument('--alpha', default=0.05, type=float, help='Strength of the adversarial loss term')
PARSER.add_argument('--beta', default=1e-5, type=float, help='Weight to use for the localizers regularization loss')
PARSER.add_argument('--segmentation_map_threshold', default=0.3, type=float,
                    help='Threshold to use when creating a segmentation map from the per-class Grad-CAMs')
PARSER.add_argument('--train_steps_localizer', default=200, type=int, help='Number of steps to train the localizer')
PARSER.add_argument('--train_steps_adversarial', default=200, type=int, help='Number of steps to train the adversarial')
PARSER.add_argument('--attention_type', default='gcam', type=str, help='Attention map method to use (cam or gcam)')
PARSER.add_argument('--seed', default=None, type=int, help='Seed for initializing training')
PARSER.add_argument('--share_weights', action='store_true', help='Share weights between the localizer and adversarial')

def main(config):
    """ Main training function """
    print(f'Running experiment with arguments:\n{config}')

    base_experiment_directory = os.path.join('results', config.experiment_name)
    if not os.path.exists(base_experiment_directory):
        os.makedirs(base_experiment_directory)

    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True

    if config.evaluate and not config.resume:
        raise ValueError('Can only evaluate from a given checkpoint, use "resume" param')

    if config.share_weights and config.adversarial_model:
        raise ValueError('When sharing weights, only the localizer needs to be set')

    # Load data
    train_loader = load_data('train', config.dataset_root, config.img_resolution, config.batch_size,
                             config.download_dataset)
    val_batch_size = 1 if config.evaluate else config.batch_size
    validation_loader = load_data('val', config.dataset_root, config.img_resolution, val_batch_size)
    print(f'Training set size: {len(train_loader.dataset)}, validation set size: {len(validation_loader.dataset)}')

    localizer = load_localizer(config.localizer_model, not config.localizer_from_scratch, config.attention_type, 
                               NUM_CLASSES)
    if torch.cuda.is_available():
        localizer = localizer.cuda()
    localizer_optimizer = load_optimizer(config.optimizer, localizer.parameters(), config.learning_rate)

    if config.adversarial_model:
        adversarial = load_adversarial(config.adversarial_model, not config.adversarial_from_scratch, NUM_CLASSES)
        if torch.cuda.is_available():
            adversarial = adversarial.cuda()
        adversarial_optimizer = load_optimizer(config.adversarial_optimizer, adversarial.parameters(),
                                               config.adversarial_learning_rate)
    else:
        adversarial = None
        adversarial_optimizer = None

    train_steps = (config.train_steps_localizer, config.train_steps_adversarial)
    training_phase = TrainingPhase(localizer, adversarial, len(train_loader), train_steps)

    start_epoch = 1

    # Load weights (from checkpoint)
    if config.resume:
        if not os.path.isfile(config.resume):
            raise Exception(f'Given checkpoint {config.resume} is not a file')
        checkpoint = torch.load(config.resume)
        localizer.load_state_dict(checkpoint['localizer'])
        localizer_optimizer.load_state_dict(checkpoint['localizer_optimizer'])
        adversarial.load_state_dict(checkpoint['adversarial'])
        adversarial_optimizer.load_state_dict(checkpoint['adversarial_optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        if config.evaluate:
            print(f'Evaluating with weights from epoch {start_epoch - 1}')
            experiment_directory = os.path.join(base_experiment_directory, str(start_epoch))
            if not os.path.exists(experiment_directory):
                os.makedirs(experiment_directory)
            validate(localizer, adversarial, validation_loader, experiment_directory, LABELS,
                     config.segmentation_map_threshold, NUM_CLASSES, evaluate=True, save_results=True)
            return

    for epoch in range(start_epoch, config.epochs + 1):
        experiment_directory = os.path.join(base_experiment_directory, str(epoch))
        if not os.path.exists(experiment_directory):
            os.makedirs(experiment_directory)
        print(f'Epoch: {epoch}/{config.epochs}')
        save_results = epoch % 3 == 0  # Save results (heatmaps, masks) every 3 epochs
        localizer.train()
        if adversarial is not None:
            adversarial.train()
        train(localizer, adversarial, localizer_optimizer, adversarial_optimizer, train_loader, epoch, config.alpha,
              config.beta, training_phase, config.share_weights)

        localizer.eval()
        if adversarial is not None:
            adversarial.eval()
        validate(localizer, adversarial, validation_loader, experiment_directory, LABELS,
                 config.segmentation_map_threshold, NUM_CLASSES, save_results=save_results)

        # Save checkpoint
        state = {
            'epoch': epoch,
            'localizer': localizer.state_dict(),
            'localizer_optimizer' : localizer_optimizer.state_dict(),
            'adversarial': adversarial.state_dict() if adversarial is not None else None,
            'adversarial_optimizer': adversarial_optimizer.state_dict() if adversarial_optimizer is not None else None
        }

        save_location = os.path.join(base_experiment_directory, 'checkpoint.pth.tar')
        torch.save(state, save_location)

def train(localizer, adversarial, localizer_optimizer, adversarial_optimizer, dataloader, epoch, alpha, beta,
          training_phase, share_weights):
    """ Loop over training set (in batches) to update the model parameters """
    localizer_criterion = torch.nn.BCELoss()
    adversarial_criterion = torch.nn.BCELoss()
    for i, (inputs, targets) in enumerate(dataloader):
        current_step = f'Epoch: {epoch}, batch: {i}\t'

        training_phase.update(epoch, i)

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        output, gcams = localizer(inputs, targets)
        gcams, new_images, new_targets, original_targets = gcams #TODO: the wrapper should just return the mapping
        masks = gcam_to_mask(gcams)

        localizer.zero_grad()
        if adversarial is not None:
            adversarial.zero_grad()

        if training_phase.train_localizer:
            loss = localizer_criterion(output, targets)
            if beta > 0:
                loss += beta * compute_regularization_loss(gcams)
            if adversarial is None:
                current_step += f'Localizer loss: {loss.item():.3f}\t'
        else:
            loss = torch.as_tensor([0])

        if adversarial is not None:
            masked_image = erase_mask(new_images, masks)
            adversarial_output = adversarial(masked_image)
            adversarial_output = torch.sigmoid(adversarial_output)
            if training_phase.train_localizer:
                am_loss = compute_am_loss(adversarial_output, new_targets)
                loss += alpha * am_loss
                current_step += f'Localizer loss: {loss.item():.3f}\t'
            if training_phase.train_adversarial:
                adversarial_loss = adversarial_criterion(adversarial_output, original_targets)
                current_step += f'Adversarial loss: {adversarial_loss.item():.3f}'

        if share_weights:
            masked_image = erase_mask(new_images, masks)
            adversarial_output = localizer(masked_image, compute_gradcam=False)
            am_loss = compute_am_loss(adversarial_output, new_targets)
            loss += alpha * am_loss
            current_step += f'Adversarial loss: {am_loss.item():.3f}'

        if i % 10 == 0:
            # Print losses every 10 steps
            print(current_step)

        if training_phase.train_localizer:
            loss.backward(retain_graph=True)
            localizer_optimizer.step()
        if training_phase.train_adversarial:
            adversarial_loss.backward()
            adversarial_optimizer.step()

def validate(localizer, adversarial, dataloader, experiment_directory, labels, segmentation_map_threshold, num_classes,
             evaluate=False, save_results=False):
    """ Loop over the validation set (in batches) to acquire relevant metrics """
    print('Validating...')
    if evaluate:
        metrics = Metrics(20)
    localizer_criterion = torch.nn.BCELoss()
    adversarial_criterion = torch.nn.BCELoss()
    localizer_loss_meter = AverageMeter()
    adversarial_loss_meter = AverageMeter()
    for i, (inputs, targets) in enumerate(dataloader):
        if evaluate:
            # Segmentation maps are included in the targets
            targets, segmentation_maps = targets
        else:
            segmentation_maps = None

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        output, gcams = localizer(inputs, labels=targets)

        loss = localizer_criterion(output, targets)
        localizer_loss_meter.update(loss.item())

        gcams, new_images, new_targets, original_targets = gcams

        if adversarial is not None or save_results:
            new_batch_size = gcams.size(0)
            masks = gcam_to_mask(gcams)

            masked_image = erase_mask(new_images, masks)
            if adversarial is not None:
                adversarial_output = adversarial(masked_image)
                adversarial_output = torch.sigmoid(adversarial_output)
                adversarial_loss = adversarial_criterion(adversarial_output, original_targets)
                adversarial_loss_meter.update(adversarial_loss.item())

            if save_results:
                for k in range(new_batch_size):
                    number = f'{i * new_batch_size + k}' #TODO: fix
                    label_string = labels[new_targets[k]]
                    file_postfix = f'{number}_{label_string}'
                    save_location = os.path.join(experiment_directory, f'heatmap_{file_postfix}.png')
                    save_gradcam(filename=save_location, gcam=gcams[k, 0].detach(), raw_image=new_images[k].clone())
                    save_location = os.path.join(experiment_directory, f'raw_heatmap_{file_postfix}.png')
                    save_gradcam(filename=save_location, gcam=gcams[k, 0].detach())
                    save_location = os.path.join(experiment_directory, f'erased_{file_postfix}.png')
                    tensor2imwrite(save_location, denormalize(masked_image[k]))

        if evaluate:
            # Generate and visualize predicted segmentation map
            predicted_segmentation_maps = generate_segmentation_map(gcams, num_classes, segmentation_maps.shape[1:], 
                                                                    new_targets, threshold=segmentation_map_threshold)
            metrics.update(predicted_segmentation_maps, segmentation_maps)

            if save_results:
                predicted_indices = predicted_segmentation_maps.unique()
                all_labels = ['background', *labels]
                predicted_labels = [all_labels[idx] for idx in predicted_indices]
                labels_string = '_'.join(predicted_labels)
                filename = f'map_{i:04d}_{labels_string}.png'
                save_location = os.path.join(experiment_directory, filename)
                save_segmentation_map(save_location, predicted_segmentation_maps, denormalize(new_images[k]).clone())
                filename = f'map_raw_{i:04d}_{labels_string}.png'
                save_location = os.path.join(experiment_directory, filename)
                save_segmentation_map(save_location, predicted_segmentation_maps)

    print('Validation localizer loss:', localizer_loss_meter.avg)
    print('Validation adversarial loss:', adversarial_loss_meter.avg)

    if evaluate:
        miou = metrics.miou().item()
        precision = metrics.precision(skip_background=True).item()
        recall = metrics.recall(skip_background=True).item()
        metrics.print_scores_per_class()
        print('mIoU:', miou)
        print('precision:', precision)
        print('recall:', recall)

if __name__ == '__main__':
    main(PARSER.parse_args())
