from tqdm import tqdm

def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion, scheduler=None, grad_clip=None):
    """
    Train for one epoch.
    - scheduler: pass your OneCycleLR here; will be stepped *per batch*
    - grad_clip: (float or None) max norm for gradient clipping (e.g., 1.0)
    """
    model.train()
    pbar = tqdm(train_loader)

    train_loss_sum = 0.0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward
        logits = model(data)
        loss = criterion(logits, target)

        # Backward
        loss.backward()

        # (Optional) gradient clipping for stability at high max_lr
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # OneCycleLR: *step per batch*
        if scheduler is not None:
            scheduler.step()

        # Book-keeping
        train_loss_sum += loss.item() * data.size(0)  # sum over samples
        correct += GetCorrectPredCount(logits, target)
        processed += data.size(0)

        # Show current LR & running stats
        curr_lr = optimizer.param_groups[0]["lr"]
        pbar.set_description(
            f"Train: loss={train_loss_sum/processed:.4f} "
            f"acc={100.0*correct/processed:.2f}% lr={curr_lr:.5f}"
        )

    # Return epoch averages (useful if you want to log outside)
    epoch_loss = train_loss_sum / processed
    epoch_acc = 100.0 * correct / processed
    return epoch_loss, epoch_acc



def test(model, device, test_loader, criterion):
    model.eval()
    test_loss_sum = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum batch loss over samples for true average at the end
            test_loss_sum += criterion(output, target).item() * target.size(0)
            correct += GetCorrectPredCount(output, target)

    avg_loss = test_loss_sum / len(test_loader.dataset)
    avg_acc = 100.0 * correct / len(test_loader.dataset)

    print(f"Test set: Average loss: {avg_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_loader.dataset)} ({avg_acc:.2f}%)\n")

    return avg_loss, avg_acc


def training_loop(model, device, train_loader, test_loader, optimizer, criterion, scheduler):
    model = Net().to(device)
    base_lr = 0.001 # Adjusted base_lr for Adam
    epochs = 16
    steps_per_epoch = len(train_loader)
    # Changed optimizer to Adam
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01, # Adjusted max_lr for Adam
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        anneal_strategy="cos",
        div_factor=10.0, # Adjusted div_factor for Adam
        final_div_factor=100.0, # Adjusted final_div_factor for Adam
        # Removed momentum parameters as they are not used in Adam
    )


    criterion = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.0)

    # Continue training from where it stopped or start fresh if needed.
    # Assuming the model and optimizer state are preserved from the previous run.
    # If you want to start training from scratch, re-initialize model and optimizer before this loop.
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        tr_loss, tr_acc = train(model, device, train_loader, optimizer, criterion, scheduler=scheduler, grad_clip=None)
        te_loss, te_acc = test(model, device, test_loader, criterion)

