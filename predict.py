import argparse

import numpy as np
import torch

from PIL import Image
from torchvision import transforms, models


def arg_parser():
    parser = argparse.ArgumentParser(description="Predict the class of a flower image")

    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the category names')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to train the model')

    return parser.parse_args()


def main():
    args = arg_parser()
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    cat_to_name = args.category_names
    device = torch.device(args.device)

    model = load_checkpoint(checkpoint)
    probs, flowers = predict(image_path, model, top_k, device, cat_to_name)

    for i in range(len(probs)):
        print(f'{flowers[i]}: {probs[i]:.2f}')


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model


def predict(image_path, model, topk=5, device='cuda', cat_to_name=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.to(device)
    model.eval()

    image = Image.open(image_path)

    image_processed = process_image(image)
    dims = np.expand_dims(image_processed, axis=0)
    normalized_image = torch.from_numpy(dims)
    normalized_image = normalized_image.type(torch.FloatTensor)
    normalized_image = normalized_image.to(device)
    props = model.forward(normalized_image)
    props = torch.exp(props)
    top_probs, top_labels = props.topk(topk)
    top_probs = top_probs.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_labels = top_labels.detach().type(torch.FloatTensor).numpy().tolist()[0]

    return top_probs, [cat_to_name[str(int(i))] for i in top_labels]


def process_image(image):
    dataloaders = transforms.Compose([transforms.RandomRotation(0),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    img = dataloaders(image).numpy()
    return torch.Tensor(img)
