import copy
from training.evaluate import evaluate_model_with_loss

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=200):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


def train_model_with_validation(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=200):
    model.train()
    epoch_accuracies = []
    epoch_losses = []
    train_epoch_losses = []

    best_loss = float('inf')
    best_model_weights = None
    patience = 10

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate validation accuracy after each epoch
        val_accuracy, val_loss = evaluate_model_with_loss(model, val_loader, criterion, device)

        epoch_accuracies.append(val_accuracy)
        epoch_losses.append(val_loss)
        train_epoch_losses.append(running_loss / len(train_loader))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here
            patience = 10  # Reset patience counter
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break


    model.load_state_dict(best_model_weights)
    return epoch_accuracies, epoch_losses, train_epoch_losses
