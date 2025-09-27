from tqdm import tqdm

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item() * len(data)  # sum up batch loss multiplied by batch size

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def training_loop(model, device, train_loader, test_loader, optimizer, criterion, scheduler):
    model = Net().to(device)
    base_lr = 0.025
    epochs = 16
    steps_per_epoch = len(train_loader)
    optimizer = optim.SGD( model.parameters(), lr=base_lr, momentum=0.9, nesterov=False )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.08,                      # peak LR you asked for
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,  # or use total_steps=epochs*steps_per_epoch
        pct_start=0.2,                    # 20% warmup
        anneal_strategy="cos",            # cosine cooldown
        div_factor=10.0,                  # initial_lr = max_lr / div_factor  -> 0.012 here (overridden by base_lr if higher)
        final_div_factor=100.0,            # min_lr = max_lr / (div_factor*final_div_factor)
        base_momentum=0.95, 
        max_momentum=0.85

    )
    criterion = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0) 
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        tr_loss, tr_acc = train(model, device, train_loader, optimizer, criterion, scheduler=scheduler, grad_clip=None)
        te_loss, te_acc = test(model, device, test_loader, criterion)

