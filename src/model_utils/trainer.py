import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def trainer(model, train_loader, value_loss_fn, policy_loss_fn, optimizer,lamda=0.5, num_epochs=10, device='mps'):
    """
    The trainer function for the model
    Args:
        model (torch.nn.Module) - the model to train
        train_loader (torch.utils.data.Dataloader) - the train data loader for the model
        value_loss_fn (torch.nn) - the loss function for the value output
        policy_loss_fn (torch.nn) - the loss function for the policy output
        optimizer (torch.optim.Optimizer) - the Optimizer of the model
        lamda (float) - the importance parameter used for the convex combination of loss functions
        num_epochs (int) - the number of training epochs, defaults to 10
        device (str) - the device to run the model, defaults to 'mps'

    Returns:
        None
    """
    if not os.path.exists("trainer"):
        os.makedirs("trainer")

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for game, value, policy in progress_bar:
            game, value, policy = (game.to(device), value.to(device), policy.to(device))

            optimizer.zero_grad()
            value_pred, policy_pred = model(game)

            value_loss = value_loss_fn(value_pred, value)
            # Flatten batch and sequence dimensions: (batch * seq_length, 128)
            policy_pred_flat = policy_pred.reshape(-1, 128)
            policy_target_flat = policy.reshape(-1, 128)

            # Create target tensor with correct shape: (batch * seq_length,)
            cosine_target = torch.ones(policy_pred_flat.shape[0], device=device)  # Ensuring correct target shape

            policy_loss = policy_loss_fn(policy_pred_flat, policy_target_flat, cosine_target)

            loss = lamda * value_loss + (1 - lamda) * policy_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        torch.save(model.state_dict(), f"trainer/model_epoch_{epoch+1}.pth")
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("trainer/training_metrics.png")
    plt.show()

