import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils import parametrize

import os
import numpy as np
from tqdm import tqdm

torch.manual_seed(42)    # for deterministic result
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 5
learning_rate = 2e-4

class MNIST_Classifer(nn.Module):
    def __init__(self, hiddien_size1=1024, hidden_size2=2048):
        super().__init__()
        self.linear1 = nn.Linear(784, hiddien_size1)
        self.linear2 = nn.Linear(hiddien_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device=torch.device("cuda")):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A)

        self.scale = alpha / rank
        self.enable = True

    def forward(self, x):
        if self.enable:
            return x + torch.matmul(self.lora_B, self.lora_A).view(x.shape) * self.scale
        else:
            return x

def layer_parameterization(layer, rank=1, alpha=1, device=torch.device("cuda")):
    features_in, features_out = layer.weight.shape
    return LoRAParametrization(features_in, features_out, rank, alpha, device)

def enable_disable_lora(net, enable=True):
    for layer in [net.linear1, net.linear2, net.linear3]:
        layer.parametrizations["weight"][0].enable = enable 

def perpare_loader(chosen_digits=None):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    if chosen_digits is not None:
        train_dataset = list(filter(lambda x: x[1] in chosen_digits, train_dataset))
        test_dataset = list(filter(lambda x: x[1] in chosen_digits, test_dataset))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

def train_one_epoch(net, loader, optimizer, criterion, max_iteration=None):
    net.train()
    loss_sum = 0
    iteration = 0
    loader = tqdm(loader, desc=f"Training")
    if max_iteration is not None:
        loader.total = min(max_iteration, loader.total)

    for images, labels in loader:
        iteration += 1
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

        loader.set_postfix(loss=loss_sum/iteration)
        
        if max_iteration is not None and iteration >= max_iteration:
            return 

def test(net, loader):
    correct = 0
    total = 0
    wrong_digits = [0] * 10
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Testing"):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            _, predicted = torch.max(output, dim=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()

            for idx, pred in enumerate(predicted):
                if pred != labels[idx]:
                    wrong_digits[labels[idx]] += 1

        print(f"Accuracy: {correct/total}")
    for idx, wrong in enumerate(wrong_digits):
        print(f"Digit {idx}: {wrong}")
    return [np.argmax(wrong_digits)]

def save_original_weights(net):
    oringinal_weights = {}
    for name, params in net.named_parameters():
        oringinal_weights[name] = params.clone().detach()
    return oringinal_weights

def freeze(net):
    for name, param in net.named_parameters():
        if 'lora' not in name:
            print(f'Freezing non-LoRA parameter {name}')
            param.requires_grad = False
        else:
            print(f'Keeping LoRA parameter {name}')

def count_params(net):
    with os.popen('stty size', 'r') as console:
        _, console_width = console.read().split()
        console_width = int(console_width)
    print("-"*console_width)
    num_params = 0
    for name, params in net.named_parameters():
        print(f"{name}: {params.shape}")
        num_params += params.numel()
    print("-"*console_width)
    return num_params 

def main():
    train_loader, test_loader = perpare_loader()
    net = MNIST_Classifer().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_one_epoch(net, train_loader, optimizer, criterion)
    chosen_digits = test(net, test_loader)
    print(f"Chosen digits: {chosen_digits}")

    original_weights = save_original_weights(net)
    total_parameters = count_params(net)

    parametrize.register_parametrization(net.linear1, "weight", layer_parameterization(net.linear1))
    parametrize.register_parametrization(net.linear2, "weight", layer_parameterization(net.linear2))
    parametrize.register_parametrization(net.linear3, "weight", layer_parameterization(net.linear3))

    total_parameters_with_lora = count_params(net)
    lora_params = total_parameters_with_lora - total_parameters
    print(f"lora % = {lora_params/total_parameters*100}%")

    train_loader, _ = perpare_loader(chosen_digits)

    freeze(net)
    # need to specify a new optimizer because the model has changed
    optimizer = optim.Adam((p for p in net.parameters() if p.requires_grad), lr=learning_rate)
    train_one_epoch(net, train_loader, optimizer, criterion)

    assert torch.all(net.linear1.parametrizations.weight.original == original_weights['linear1.weight'])
    assert torch.all(net.linear2.parametrizations.weight.original == original_weights['linear2.weight'])
    assert torch.all(net.linear3.parametrizations.weight.original == original_weights['linear3.weight'])

    enable_disable_lora(net, enable=True)
    assert torch.equal(net.linear1.weight, net.linear1.parametrizations.weight.original + (net.linear1.parametrizations.weight[0].lora_B @ net.linear1.parametrizations.weight[0].lora_A) * net.linear1.parametrizations.weight[0].scale)
    test(net, test_loader)

    enable_disable_lora(net, enable=False)
    assert torch.equal(net.linear1.weight, original_weights['linear1.weight'])
    test(net, test_loader)


if __name__ == "__main__":
    main()