import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from src.model import resnet20

def load_model_and_test(model, checkpoint_path, test_loader, device):
    """
    Load model from checkpoint and test it on the testing dataset.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['net'])
    model.to(device)

    # Test model
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    test_accuracy = 100 * correct / total
    return test_accuracy


if __name__ == '__main__':
    checkpoint_path = sys.argv[1]
    model = resnet20()

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # Transform for test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    #  Load test set
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # Load model and test on test set
    test_accuracy = load_model_and_test(model, checkpoint_path, testloader, device)
    print(f'Model Accuracy on Training Set: {test_accuracy:.3f}%')