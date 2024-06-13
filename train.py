# Imports here

import torch
from torch import nn
from torch import optim

from torchvision import datasets, transforms, models

from collections import OrderedDict

import argparse
import json


def arg_parser():
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument('data_dir', type=str, help='Directory containing the data')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the model')
    parser.add_argument('--arch', type=str, default='vgg19', help='Architecture of the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--print_every', type=int, default=5, help='Print every n steps')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to train the model')
    return parser.parse_args()


def main():
    args = arg_parser()
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    print_every = args.print_every
    epochs = args.epochs
    dropout = args.dropout
    batch_size = args.batch_size
    device = torch.device(args.device)
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    image_datasets = transforms.Compose([transforms.RandomRotation(255),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomVerticalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=image_datasets)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size / 2)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = 25088
    else:
        print('Architecture not supported')
        return

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units, bias=True)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=dropout)),
        ('fc2', nn.Linear(hidden_units, 256, bias=True)),
        ('fc3', nn.Linear(256, 102, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)

    steps = 0
    running_loss = 0

    for epoch in range(epochs):

        for images, labels in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(images)

            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        loss = criterion(logps, labels)

                        test_loss += loss.item()

                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {test_loss / len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy / len(validloader):.3f}")
                running_loss = 0
                model.train()

        checkpoint = {'input_size': input_size,
                      'output_size': 102,
                      'classifier': model.classifier,
                      'state_dict': model.state_dict()}
        torch.save(checkpoint, save_dir + '/checkpoint.pth')


print("Training concluded")

if __name__ == '__main__':
    main()
