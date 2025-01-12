import torch
import torch.optim as optim
import torch.nn as nn

def fine_tune_and_evaluate_qat(model, train_loader, test_loader, device="cpu",
                               epochs=2, lr=1e-4, noise_ratio=0):
    model = model.to(device)
    model.eval()

    # 1) Fuse
    model.fuse_model()
    model.train()
    # 2) QConfig
    model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")

    # 3) Prepare QAT
    torch.quantization.prepare_qat(model, inplace=True)

    # 4) Train (fine-tune)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[QAT Train] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # 5) Convert to final quantized model
    model.eval()
    quantized_model = torch.quantization.convert(model.to("cpu"), inplace=False)
    torch.save(quantized_model.state_dict(), f"quantized_model{noise_ratio}.pth")
    # 6) Evaluate quantized model (on CPU)
    quantized_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to("cpu"), targets.to("cpu")
            outputs = quantized_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    print(f"[Quantized Eval] Accuracy: {accuracy:.2f}%")
    return quantized_model, accuracy
