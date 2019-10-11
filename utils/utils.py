import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
#from torch.utils.data.sampler import SubsetRandomSampler


def train_validation_split(dataset, fraction=0.1, batchsize=64):
    """
    @ dataset: torvision dataset
    return: training and validation set as DataLoaders
    """
    n_val_images = int(len(dataset) * fraction)
    val_indices = choice(range(len(dataset)), size=n_val_images, replace=False)
    train_indices = [x for x in range(len(dataset)) if x not in val_indices]

    train_loader = DataLoader(Subset(dataset, train_indices),
                              #sampler=SubsetRandomSampler(train_indices),
                              shuffle=True,
                              batch_size=batchsize
                              )
    val_loader = DataLoader(Subset(dataset, val_indices),
                            #sampler=SubsetRandomSampler(val_indices),
                            shuffle=False,
                            batch_size=batchsize
                            )
    return train_loader, val_loader


def prediction_from_output(model_output: torch.Tensor):
    """
    @ model_output: output from the last linear layer (the output is without
                    softmax because softmax is included in CE-loss)
    return: predictions as tensor of class-indices
    """
    probabilities = nn.Softmax(dim=1)(model_output)
    max_probs, predictions = probabilities.max(dim=1)  
    return max_probs, predictions
  

def train_model(net, train, validation, optimizer, scheduler, max_epoch=100):
    """
    This function returns nothing. The parametes of @net are updated in-place
    and the error statistics are written to a global variable. This allows to
    stop the training at any point and still have the results.
  
    @ net: a defined model - can also be pretrained
    @ train, test: DataLoaders of training- and test-set
    @ max_epoch: stop training after this number of epochs
    """  
    global error_stats  # to track error log even when training aborted
    error_stats = []
  
    criterion = nn.CrossEntropyLoss()
    net.cuda()
    min_error = np.inf
  
    print('epoch\ttraining-CE\tvalidation-CE\tvalidation-accuracy (%)')
    for epoch in range(max_epoch):
        net.train()
        training_loss = 0
    
        for images, labels in train:
            labels = labels.cuda()
            images = images.cuda()
            optimizer.zero_grad()

            # prediction and error:
            output = net(images)
            loss = criterion(output, labels)  # loss of current batch
            training_loss += loss.item() / len(train)

            # update parameters:
            loss.backward()
            optimizer.step()

        with torch.no_grad():  # no backpropagation necessary
            validation_loss = 0
            net.eval()

            for images, labels in validation:
                labels = labels.cuda()
                images = images.cuda()

                # prediction and error:
                output = net(images)
                loss = criterion(output, labels)
                validation_loss += loss.item() / len(validation)

                predictions = prediction_from_output(output)[1]
                accuracy = (predictions == labels).float().mean() * 100
    

        scheduler.step(validation_loss)

        if validation_loss < min_error:
            torch.save(net.state_dict(), f'{net.__class__.__name__}_best.pt')
            min_error = validation_loss
            
        error_stats.append( (training_loss, validation_loss) )
        print('{}\t{:.2f}\t\t{:.2f}\t\t{:.2f}'.format(
            epoch, training_loss, validation_loss, accuracy)
             )

    
def test_set_evaluation(net, test, just_print=False):
    """
    Calculate cross-entropy loss (mean batch loss) and accuracy on the test-set
    """
    global accuracies
    total_loss = 0
    net.eval()
    criterion = nn.CrossEntropyLoss()
    accuracies = []
  
    with torch.no_grad():
        for images, labels in test:
            labels = labels.cuda()
            images = images.cuda()

            output = net(images)
            batch_loss = criterion(output, labels)
            total_loss += batch_loss.item()

            predictions = prediction_from_output(output)[1]
            batch_accuracy = (predictions == labels).float().mean() * 100
            accuracies.append(batch_accuracy.item())
    
    mean_batch_loss = total_loss / len(test)
    accuracy = np.mean(accuracies)
  
    if just_print:
        print('\nEvaluation on the test-set:')
        print(f'mean batch cross-entropy loss: {mean_batch_loss:.2f}')
        print(f'accuracy: {accuracy:.2f}')
        return None
  
    return mean_batch_loss, accuracy
    

def count_parameters(model, in_millions=False):
    """
    Count number of parameters of @model
    """
    n_params = sum(p.numel() for p in model.parameters())
    if in_millions:
        n_params = n_params / 1000000
    return n_params


def show_image(processed_image: torch.Tensor, means: tuple, stdevs: tuple):
    """
    @ means / @ stdevs: per-channel values used for normalizing the raw images
    Recalculate original image from @processed_image and display it.
    """
    img = processed_image.clone()
    for channel in (0, 1, 2):
        img[channel, :, :] = img[channel, :, :] * stdevs[channel] + means[channel]
    img = img.permute(1, 2, 0).numpy()

    plt.imshow(img)
    plt.grid('off')
    plt.xticks([])
    plt.yticks([])
    plt.show();


def class_name(index: int) -> str:
    """
    @ index: class-index between 0-99
    return: class-name 
    """
    class_names = [
      'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
      'beetle','bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly',
      'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
      'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
      'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl',
      'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower',
      'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
      'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
      'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
      'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
      'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper',
      'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
      'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
      'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
      'willow_tree', 'wolf', 'woman', 'worm'
    ]
    class_index2name = dict(enumerate(class_names))
    return class_index2name.get(index, f'invalid class index: {index}')


def predict_and_display(net,
                        testset,
                        n=10,
                        channel_means=(0.5071, 0.4865, 0.4409),  # CIFAR-100 values
                        channel_standard_devs=(0.2673, 0.2564, 0.2762)):
    """
    Predict @n random examples from the test-set and show images + predictions
    """
    net.eval()
    for i in choice(range(len(testset)), size=n):
        image, label = testset[i]
        output = net(image.unsqueeze(0).cuda())
        prob, pred = prediction_from_output(output.unsqueeze(0))
        prob, pred = prob.item(), pred.item()
        evaluation = 'correct' if pred == label else 'mistake'

        plt.figure( figsize=(2, 2) )
        print(f'\ntruth: {label} | pred: {pred} | prob: {prob:.2f}')
        print(f'{evaluation}: ({class_name(label)} vs. {class_name(pred)})')
        show_image(image, means=channel_means, stdevs=channel_standard_devs)


def plot_error_curves(errors_over_time: list, error_name='error'):
    """
    @ errors_over_time: list of tuples: (training-error, validation-error)
    """
    error_train, error_validation = zip(*errors_over_time)

    plt.plot(range(len(error_train)), error_train)
    plt.plot(range(len(error_validation)), error_validation)
    plt.xticks(range(0, len(error_train) + 1, len(error_train) // 2))
    plt.xlabel('epoch')
    plt.ylabel('CE')
    plt.legend(('training', 'validation'))
    plt.title(f'{error_name} over time')
    plt.show();