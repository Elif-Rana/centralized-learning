import argparse
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import mobilenet_v3_small
from tqdm import tqdm
from datetime import datetime

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Centralized Learning")
parser.add_argument(
    "--epochs",
    default=3,
    type=int,
    required=False,
    help="Number of epochs"
)

def train(net,trainloader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    #Define optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs} ...")
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            net.to(DEVICE)
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()

def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0,0,0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)

            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    loss = loss / len(testloader.dataset)
    accuracy = correct / total

    return loss, accuracy

def load_data():
    trf = Compose([ToTensor(),Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Split the data
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=16, shuffle=True), DataLoader(testset, batch_size=64)

def load_model():
    return mobilenet_v3_small(num_classes=10)

def save_model(net, accuracy, epochs,  model_filename="model", accuracy_filename="accuracy.txt"):

    # Generate a timestamp to identify models and their accuracy values
    timestamp = datetime.now().strftime("%d.%m.%Y_%H.%M")

    model_filename = f"{model_filename}_{timestamp}.pth"

    # Save the model and accuracy and epoch values

    with open(accuracy_filename, "a") as f: # "a" is for append
        f.write(f"{timestamp}: Model Filename: {model_filename} - Accuracy: {accuracy:.3f} - Epochs: {epochs}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    net = load_model()
    trainloader, testloader = load_data()

    train(net, trainloader, epochs=args.epochs)
    loss, accuracy = test(net, testloader)

    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")
    save_model(net, accuracy, args.epochs)
