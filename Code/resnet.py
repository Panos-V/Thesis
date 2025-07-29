import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import SkinCancerData
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy(force = True)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    progress_bar = tqdm(train_loader,desc = f"Epoch {epoch+1}/{num_epochs}", unit="batch")
                    model.train()  # Set model to training mode
                else:
                    print("Evaluating on validation set...")
                    progress_bar = tqdm(test_loader,desc = f"Epoch {epoch+1}/{num_epochs}", unit="batch")
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                all_labels = []
                all_preds = []

                # Iterate over data.
                for inputs, labels, _ in progress_bar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"},refresh=True)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / (train_loader.dataset.__len__() if phase == 'train' else test_loader.dataset.__len__())
                epoch_acc = running_corrects.double() / (train_loader.dataset.__len__() if phase == 'train' else test_loader.dataset.__len__())

                # Compute metrics
                acc = accuracy_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds, average='weighted')
                precision = precision_score(all_labels, all_preds, average='weighted')
                recall = recall_score(all_labels, all_preds, average='weighted')

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                print(f'{phase} Classification Report:')
                print(f'Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}')
                print(classification_report(all_labels, all_preds, digits=4))
                print("Confusion Matrix:",confusion_matrix(all_labels, all_preds))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model

def visualize_model(model, loader, num_images=32):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(16, 16))  # Adjust size as needed

    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(8, 4, images_so_far)  # 8 rows, 4 columns = 32 images
                ax.axis('off')
                ax.set_title(f'pred: {preds[j]}\nactual: {labels[j]}', fontsize=8)
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.savefig('predictions.png')
                    return

def run():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256))
        ])
    skin_train, skin_test = SkinCancerData.CreateLoader("Code/archive/", transform, batch_size=64)

    res = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in res.parameters():
        param.requires_grad = False
    res.fc = nn.Linear(res.fc.in_features, 2)
    res.to(device)
    res_criterion = nn.CrossEntropyLoss()
    res_optimizer = optim.SGD(res.parameters(), lr=1e-3,momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(res_optimizer, step_size=5, gamma=0.1)

    res = train_model(res, res_criterion, res_optimizer, scheduler, skin_train, skin_test, num_epochs=5)
    torch.save(res.state_dict(), "best_resnet20.pth")
    """ res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    res.fc = nn.Linear(res.fc.in_features, 2)  # Make sure this matches your training setup
    res.load_state_dict(torch.load("best_resnet50.pth", map_location=device))
    res.to(device) """

    visualize_model(res,skin_test, num_images=32)