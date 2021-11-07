import time
import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # 在数据处理里已经做了标准化了，所以这里不再需要做了
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)

        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input_ = self.next_input
        target = self.next_target
        self.preload()
        return input_, target

def train(model, train_loader, device, loss_fn, optimizer, train_total, epoch):

    model.train()

    train_loss = 0.
    train_acc = 0.
    total = 0
    for i, (images, labels) in enumerate(train_loader):

        # Forward
        images = images['image'].to(device)
        labels = labels.to(device).long()
        preds = model(images)

        # Compute loss
        loss = loss_fn(preds, labels) 
        train_loss += loss.item()
     
        # Backward        
        optimizer.zero_grad()                     
        loss.backward()

        # Update weights
        optimizer.step()

        # Prediction -> acc
        _, pred_labels = torch.max(preds, 1)
        # pred_labels = preds.squeeze()
        batch_correct = (pred_labels==labels).squeeze().sum().item()
        train_acc += batch_correct

        batch_size = labels.size(0)
        total += batch_size
    train_acc = train_acc / train_total
    train_loss = train_loss / len(train_loader)
    
    return train_acc, train_loss
    
def valid(model, val_loader, device, loss_fn, optimizer, valid_total, epoch):
    valid_acc = 0.
    valid_loss = 0.
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):

            images = images['image'].to(device)
            labels = labels.to(device).long()

            preds = model(images)

            loss = loss_fn(preds, labels)
            valid_loss += loss.item()

            _, pred_labels = torch.max(preds, 1)
            batch_correct = (pred_labels==labels).squeeze().sum().item()
            valid_acc += batch_correct
            
    valid_acc = valid_acc / valid_total
    valid_loss = valid_loss / len(val_loader)

    return valid_acc, valid_loss
