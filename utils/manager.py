import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CyclicLR, OneCycleLR, StepLR
from torchvision import transforms

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
import time

from sklearn.metrics import confusion_matrix, classification_report

class ModelManager:
    def __init__(self, model, optimizer, loss_fn, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            if device=='cuda' and not torch.cuda.is_available():
                print('cuda is not available. Device will be set to `cpu`')
                self.device = 'cpu'
            
        self.model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self.model.to(self.device)
        
        self._train_losses = []
        self._val_losses = []
        self.lrs = set()
        
        self._train_data = None
        self._val_data = None
        self._train_step = self._train_step_fn()
        self._val_step = self._val_step_fn()

        self._total_epochs = 0
        
        self._lr_scheduler = None
        self._STEP_SCHEDULER = False
        self._BATCH_SCHEDULER = False

        self._filename = None
        
    def _train_step_fn(self):
        def _step(x, y):
            self.model.train()
            
            x, y = x.to(self.device), y.to(self.device)

            yhat = self.model(x)
            loss = self._loss_fn(yhat, y)
            
            loss.backward()
            
            self._optimizer.step()
            self._optimizer.zero_grad()
            
            if self._BATCH_SCHEDULER:
                self._lr_scheduler.step()
    
            return loss.item(), self._accuracy(x, y)
        return _step

    def _val_step_fn(self):
        def _step(x, y):
            self.model.eval()
            x, y = x.to(self.device), y.to(self.device)
            yhat = self.model(x)

            loss = self._loss_fn(yhat, y)
            return loss.item(), self._accuracy(x, y)
        return _step

    def _mini_batch(self, validation=False):
        if validation:
            dataloader = self._val_data
            step_fn = self._val_step
        else:
            dataloader = self._train_data
            step_fn = self._train_step
            
        loss, accuracy = 0, 0
        for images, labels in dataloader:
            loss_batch, accuracy_batch = step_fn(x=images, y=labels)
            loss += loss_batch
            accuracy += accuracy_batch
            
        return loss/len(dataloader), accuracy/len(dataloader)

    
    def set_lr_scheduler(self, scheduler):
        if scheduler.optimizer != self._optimizer:
            raise ValueError('Optimizer is not used in lr_scheduler')
        self._lr_scheduler = scheduler
        if isinstance(scheduler, StepLR) or \
            isinstance(scheduler, MultiStepLR) or \
            isinstance(scheduler, ReduceLROnPlateau):
            self._STEP_SCHEDULER = True
        elif isinstance(scheduler, CyclicLR) or isinstance(scheduler, OneCycleLR):
            self._BATCH_SCHEDULER = True

    
    def train(self, epochs, seed=42, print_loss=False):
        self._set_seed(seed)
        last_loss = None
        
        for epoch in tqdm(range(epochs)):
            try:
                self._total_epochs += 1
                loss, train_acc = self._mini_batch()
                self._train_losses.append(loss)
    
                with torch.no_grad():
                    val_loss, val_acc = self._mini_batch(validation=True)
                    self._val_losses.append(val_loss)
    
                if self._STEP_SCHEDULER:
                    if isinstance(self._lr_scheduler, ReduceLROnPlateau):
                        self._lr_scheduler.step(val_loss)
                    else:
                        self._lr_scheduler.step()
                    if self._lr_scheduler.optimizer.param_groups[0]['lr'] not in self.lrs:
                        self.lrs.add(self._lr_scheduler.optimizer.param_groups[0]['lr'])
                        print('learning rate changed to ---> ', self._lr_scheduler.optimizer.param_groups[0]['lr'])
    
                
                if last_loss is None or last_loss > val_loss:
                    if self._filename is not None:
                        self.save_checkpoint('best_'+self._filename)
                    else:
                        self.save_checkpoint('best')
                        
                if print_loss:
                    print(f'Epoch {self._total_epochs}| train Loss: {loss:0.4f} | acc. {train_acc:0.2f}% \
                    | val Loss: {val_loss:0.4f} | val acc. {val_acc:0.2f}%')
                    
            except KeyboardInterrupt:
                if self._filename is not None:
                    self.save_checkpoint('last_'+self._filename)
                else:
                    self.save_checkpoint('last')
                raise KeyboardInterrupt('Keyboard Interrupt')

            if self._filename is not None:
                self.save_checkpoint('last_'+self._filename)
            else:
                self.save_checkpoint('last')
    
    def set_dataloaders(self, train_data, val_data=None):
        self._train_data= train_data
        if val_data is not None:
            self._val_data = val_data

    
    def _accuracy(self, x, target):
        self.model.eval()
        output = self.model(x.to(self.device))
        # Get predicted class (argmax across class dimension)
        predictions = torch.argmax(output, dim=1)
    
        # Compare predictions with ground truth
        correct = (predictions == target).sum().item()
    
        # Calculate accuracy
        accuracy = correct / target.size(0) * 100
        return accuracy


    def _set_seed(self, seed):
        if self.device=='cuda':
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)

    def predict(self, x):
        self.model.eval()
    
        x_tensor = x.unsqueeze(0)
        pred = self.model(x_tensor.to(self.device))
        prop = F.softmax(pred, dim=1)
        pred_class = prop.argmax(dim=1).detach().cpu().numpy()
        self.model.train()
        return pred_class.item()
        
    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self._total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self._optimizer.state_dict(),
                      'loss': self._train_losses,
                      'val_loss': self._val_losses}
    
        torch.save(checkpoint, filename)
        
    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename, map_location=torch.device(self.device))
    
        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
        self._total_epochs = checkpoint['epoch']
        self._train_losses = checkpoint['loss']
        self._val_losses = checkpoint['val_loss']
    
        self.model.train()

    def set_filename(self, filename):
        self._filename = filename
        
    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self._train_losses, label='Training Loss', c='b')
        plt.plot(self._val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig
        
    def conf_mat_class_report(self, classes, validation=False, report=False):
        torch.manual_seed(42)
        if self._filename is None:
            self.load_checkpoint('best')
        else:
            self.load_checkpoint('best_'+self._filename)
        self.model.eval()
        y_pred = []
        y_true = []
        if validation:
          dataloader = self._val_data
        else:
          dataloader = self._train_data
        # iterate over test data
        with torch.inference_mode():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                output = self.model(inputs) # Feed Network
        
                output = output.argmax(dim=1).cpu().numpy()
                y_pred.extend(output) # Save Prediction
        
                labels = labels.cpu().numpy()
                y_true.extend(labels) # Save Truth
        
        self.model.train()
        
        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix # / np.sum(cf_matrix, axis=1)[:, None] #to print the precentage
                             , index = [i for i in classes],
                             columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True, fmt='')
        if report:
          print(classification_report(y_true, y_pred))
        
    
    def printNetResults(self, img, label_mapping):
        img = img.unsqueeze(0)
        if self._filename is None:
            self.load_checkpoint('best')
        else:
            self.load_checkpoint('best_'+self._filename)
        self.model.eval()
        start_time = time.perf_counter()
        yHat = self.model(img.to(self.device))
        end_time = time.perf_counter()
        self.model.train()
        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"Forward pass took {elapsed_time_ms:.2f} ms")
        
        probabilities = F.softmax(yHat, dim=1)
        
        for i in range(4):
            class_label = label_mapping[i]
            prob_percent = 100.*probabilities[0][i]
            print( f"Class {i+1}: {class_label} with probability {prob_percent:.2f}%" )


def normalize(dataloader):
    mean = torch.tensor([0, 0, 0], dtype=torch.float)
    std = torch.tensor([0, 0, 0], dtype=torch.float)

    for img, _ in tqdm(dataloader):
        n_samples, n_channel, h, w = img.size()

        value_per_channel = img.reshape(n_samples, n_channel, h*w)
        means = value_per_channel.mean(axis=2)
        stds = value_per_channel.std(axis=2)

        mean += means.mean(axis=0)
        std += stds.mean(axis=0)
    mean /= len(dataloader)
    std /= len(dataloader)
    
    return transforms.Normalize(mean=mean,std= std)

def denormalize(img, normalizer):
    mean = normalizer.mean
    std = normalizer.std
    inv_normalize = transforms.Normalize(
                  mean= [-m/s for m, s in zip(mean, std)],
                  std= [1/s for s in std]
                  )
    return inv_normalize(img)
    
def show_samples(n, images, b_labels, class_labels, normalizer=None, denormalizer=None):
    c = 4 if n >= 4 else n
    r = n//c if n%c == 0 else (n//c)+1

    fig = plt.figure(figsize=(10, c*2))
    for i, img in enumerate(images):
        if i == n:
            break
        ax = fig.add_subplot(r, c, i+1)
        ax.set_xlabel(class_labels[b_labels[i]])

        if denormalizer is not None:
            img = denormalize(img, normalizer)
        img = np.clip(img, 0, 1)

        plt.imshow(np.transpose(img.numpy(), (2, 1, 0)))