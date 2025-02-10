import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init

def evaluate_model(model, val_loader, device):
    """
    Evaluates the model on the validation set and prints accuracy.
    Returns the accuracy value for checkpoint tracking.
    """
    model.eval()  # Set to evaluation mode
    correct, total = 0, 0

    with torch.no_grad():  # No gradients needed during evaluation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)  # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path="best_model.pth",
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Trains the model for a specified number of epochs.
    Saves the best model checkpoint (based on validation accuracy).
    """
    model = model.to(device)  # Move model to the correct device
    best_val_accuracy = 0.0  # Track the highest validation accuracy
    train_loss_history = []

    for epoch in range(num_epochs):
        model.train()  # Set to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Ensure data is on the same device

            optimizer.zero_grad()          # 1) Zero out previous gradients
            outputs = model(images)        # 2) Forward pass
            loss = criterion(outputs, labels)  # 3) Compute loss
            loss.backward()                # 4) Backpropagation
            optimizer.step()               # 5) Update parameters

            running_loss += loss.item()

        # Compute average training loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        train_loss_history.append(epoch_loss)

        print(f"[Epoch {epoch+1:02d}/{num_epochs:02d}] "
              f"Train Loss: {epoch_loss:.4f}")

        # Evaluate on validation set
        val_accuracy = evaluate_model(model, val_loader, device)

        # Check if current model is the best so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved! New best accuracy: {best_val_accuracy:.2f}%\n")

    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.2f}%")
    return train_loss_history

def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    """Initialize layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')  # Fixed kaiming_normal to kaiming_normal_
            if m.bias is not None:  # Fixed bias condition check
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)  # Fixed normal to normal_
            if m.bias is not None:  # Fixed bias condition check
                init.constant_(m.bias, 0)

term_width = 80
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return 