import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from model import resnet20


def parse_args(args):
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["SGD", "Adam", "AdamW"], help="Optimizer type")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--step_size", type=int, default=40, help="Step size for learning rate scheduler")
    parser.add_argument("--augment", type=bool, default=False, help="Augment data")

    args = parser.parse_args(args)
    return args


def main(args):
    # Get arguments
    args = parse_args(args)

    # Initialize Weights & Biases
    wandb.init(
        project="Lab5-ResNet20", 
        name=f"resnet20_lr_{args.lr}_batch_{args.batch_size}_epochs_{args.epochs}_optimizer_{args.optimizer}_weight_decay_{args.weight_decay}_augment_{args.augment}", 
        config={           
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "step_size": args.step_size,
            "augment": args.augment,
        }
    )
    config = wandb.config

    # Data pre-processing
    print('==> Preparing data..')
    if config.augment:  # Augment data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add color jitter
            transforms.RandomRotation(15),  # Random rotation
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Get training data
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)

    # Get testing data
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False)

    classes = ("airplane", "automobile", "bird", "cat",
            "deer", "dog", "frog", "horse", "ship", "truck")

    # Model
    print('==> Building model..')
    model = resnet20()

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (AdamW by default)
    if config.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Learning rate scheduler (StepLR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=0.1)
    
    # Training for one epoch
    def train(epoch):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print('Epoch [%d] Batch [%d/%d] Loss: %.3f | Training Acc: %.3f%% (%d/%d)'
                % (epoch, batch_idx + 1, len(trainloader), train_loss / (batch_idx + 1),
                    100. * correct / total, correct, total))

        avg_train_loss = train_loss / len(trainloader)
        train_accuracy = 100. * correct / total

        return avg_train_loss, train_accuracy

    # Testing for one epoch
    def test(epoch):
        print('==> Testing...')
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # Calculate testing accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_accuracy = 100 * correct / total
        print(f"Epoch [{epoch}] - Test Accuracy: {test_accuracy:.3f}%")

        # Save checkpoint
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': test_accuracy,
            'epoch': epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/ckpt_{epoch}_acc_{test_accuracy:.3f}.pth')

        return test_accuracy


    # Training and Testing loop
    for epoch in range(1, config.epochs + 1):
        avg_train_loss, train_accuracy = train(epoch)
        test_accuracy = test(epoch)

        wandb.log({
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
        })

        scheduler.step()

    wandb.finish()


if __name__ == '__main__':
    main(sys.argv[1:])