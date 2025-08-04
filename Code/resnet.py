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

def fairness_metrics(y_true, y_pred, protected):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    protected = np.array(protected)

    protected_mask = protected == 1
    non_protected_mask = protected == 0

    # EO: TPR ratio
    positive_true_mask = y_true == 1
    tp_protected = np.sum((positive_true_mask & (y_pred == 1) & protected_mask))
    tp_non_protected = np.sum((positive_true_mask & (y_pred == 1) & non_protected_mask))
    pos_protected = np.sum(positive_true_mask & protected_mask)
    pos_non_protected = np.sum(positive_true_mask & non_protected_mask)
    EO = (tp_protected / pos_protected) / (tp_non_protected / pos_non_protected) if pos_protected > 0 and pos_non_protected > 0 else np.nan

    # DI: Success ratio (TP + TN)
    success_protected = np.sum(((y_true == y_pred) & protected_mask))
    success_non_protected = np.sum(((y_true == y_pred) & non_protected_mask))
    total_protected = np.sum(protected_mask)
    total_non_protected = np.sum(non_protected_mask)
    DI = (success_protected / total_protected) / (success_non_protected / total_non_protected) if total_protected > 0 and total_non_protected > 0 else np.nan

    return EO, DI


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy(force = True)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def train_model(model,vq,classifier, criterion, optimizer, train_loader, test_loader,adversarial = False ,ALPHA = 0.025,num_epochs=25):
    since = time.time()
    vq.eval()
    classifier.eval()
    vq.to(device)
    classifier.to(device)
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
                all_protected = []

                # Iterate over data.
                for inputs, labels, protected in progress_bar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if adversarial_walk and phase == 'train':
                            h = vq.encoder(inputs)
                            h = vq.pre_vq_conv(h)
                            output,perplexity = adversarial_walk(classifier, h, ALPHA,vq,device)
                            recon = vq.decoder(output).to(device)
                            outputs = model(recon)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                        elif phase == 'train':
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
                        else:
                            with torch.no_grad():
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
                    all_protected.extend(protected.cpu().numpy())
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"},refresh=True)

                epoch_loss = running_loss / (train_loader.dataset.__len__() if phase == 'train' else test_loader.dataset.__len__())
                epoch_acc = running_corrects.double() / (train_loader.dataset.__len__() if phase == 'train' else test_loader.dataset.__len__())

                # Compute metrics
                acc = accuracy_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds, average='weighted')
                precision = precision_score(all_labels, all_preds, average='weighted')
                recall = recall_score(all_labels, all_preds, average='weighted')
                EO, DI = fairness_metrics(all_labels, all_preds, all_protected)
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                print(f'{phase} Classification Report:')
                print(f'Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}')
                print(classification_report(all_labels, all_preds, digits=4))
                print("Confusion Matrix:",confusion_matrix(all_labels, all_preds))
                print(f'Fairness Metrics - Equal Opportunity: {EO:.4f}, Impact Disparate: {DI:.4f}')
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

def inference(model_path,vq,classifier, loader, transform,adversarial=False, ALPHA=0.025):
    # Load model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_protected = []

    progress_bar = tqdm(loader, desc="Inference", unit="batch")

    for inputs, labels, protected in progress_bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if adversarial:
            # DO NOT use torch.no_grad() here!
            h = vq.encoder(inputs)
            h = vq.pre_vq_conv(h)
            output, _ = adversarial_walk(classifier, h, ALPHA, vq, device)
            recon = vq.decoder(output).to(device)
            inputs = recon
            with torch.no_grad():
                outputs = model(inputs)
        else:
            with torch.no_grad():
                outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_protected.extend(protected.cpu().numpy())

        progress_bar.set_postfix({"Accuracy": f"{accuracy_score(all_labels, all_preds):.4f}"}, refresh=True)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    EO, DI = fairness_metrics(all_labels, all_preds, all_protected)
    print(f'Inference Classification Report:')
    print(f'Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}')
    print(classification_report(all_labels, all_preds, digits=4))
    print("Confusion Matrix:",confusion_matrix(all_labels, all_preds))
    print(f'Fairness Metrics - Equal Opportunity: {EO:.4f}, Impact Disparate: {DI:.4f}')

def adversarial_walk(f,h,a,model,device,steps = 4):    #h = latent representations f = classifier
    h_delta = h.clone().detach().requires_grad_(True).to(device)

    e = 1e-12
    for i in range(steps):
        prediction = f(h_delta)
        prediction = torch.softmax(prediction, dim=1)
        entropy = -torch.special.entr(prediction + e).sum(dim=1).mean()
        gradient = torch.autograd.grad(entropy, h_delta, retain_graph=True)[0]


        delta = (gradient - gradient.mean()) / (gradient.std() + e)    

        h_delta = h_delta + a*delta

        _,h_delta,perplexity,_ = model.vq(h_delta)

        
        h_delta = h_delta.requires_grad_(True)

    #print(h_delta)

    return h_delta,perplexity

def fine_tune(model, criterion, optimizer, train_loader, test_loader, num_epochs=100, patience=5, min_delta=0.01):
    since = time.time()
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        epochs_no_improve = 0

        for param in model.parameters():
            param.requires_grad = True

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    loader = train_loader
                else:
                    model.eval()
                    loader = test_loader

                running_loss = 0.0
                running_corrects = 0
                all_labels = []
                all_preds = []
                all_protected = []

                for inputs, labels, protected in loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_protected.extend(protected.cpu().numpy())

                epoch_loss = running_loss / len(loader.dataset)
                epoch_acc = running_corrects.double() / len(loader.dataset)

                if phase == 'val':
                    if epoch_acc > best_acc + min_delta:
                        best_acc = epoch_acc
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    EO, DI = fairness_metrics(all_labels, all_preds, all_protected)
                    print(f'Fairness Metrics - Equal Opportunity: {EO:.4f}, Impact Disparate: {DI:.4f}')
                
        
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model

def create_model(vq,classifier,train,test,name = 'ResNet18.pth',adversarial = False, ALPHA = 0.025):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256))
        ])

    res = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in res.parameters():
        param.requires_grad = False
    res.fc = nn.Linear(res.fc.in_features, 2)
    res.to(device)
    res_criterion = nn.CrossEntropyLoss()
    res_optimizer = optim.SGD(res.parameters(), lr=1e-3,momentum=0.9)


    res = train_model(res,vq,classifier, res_criterion, res_optimizer, train, test, num_epochs=5,adversarial=adversarial, ALPHA=ALPHA)
    res = fine_tune(res, res_criterion, res_optimizer, train, test, num_epochs=100, patience=10, min_delta=0.01)
    torch.save(res.state_dict(), name)

    visualize_model(res,test, num_images=32)

    return res

""" transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256))
    ])
_,skin_test = SkinCancerData.CreateLoader("Code/archive/", transform, batch_size=64) """
#inference("bias_resnet20.pth", skin_test,transform)