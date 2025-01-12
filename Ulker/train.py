import torch
import torch.nn as nn
import torch.optim as optim
from src.RCM import RCM
from src.dataset_loader import CustomDataset, get_dataloaders
from torchsummary import summary

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for img1, img2, labels in train_dataloader:
            img1, img2 = img1.to(device), img2.to(device)

            # Etiketler
            keypoints0 = labels["keypoints0"].to(device)
            keypoints1 = labels["keypoints1"].to(device)
            matches = labels["matches"].to(device)
            match_confidence = labels["match_confidence"].to(device)

            # convert non 1024 sized matches to 1024 size
            matches_converted = torch.ones((1024)) * -1
            matches_converted[0:len(matches[0])] = matches

            # Sıfırla gradyanları
            optimizer.zero_grad()

            # Modelden geçirin
            outputs = model(img1, img2)

            # Kayıp hesaplama (örnek olarak keypoints0 ve outputs arasında)
            loss = criterion(outputs, torch.tensor(matches_converted, dtype=torch.float32, device=device))
            
            # Geri yayılım ve optimizasyon
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        val_avg = validate_model(model, val_dataloader, criterion, device)
        if ((epoch + 1) % 5 == 0):
            # Modeli kaydet
            torch.save(model.state_dict(), f"weights_{epoch + 1}.pth")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss / len(train_dataloader):.4f}, Validation Loss: {val_avg}")


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1, img2 = img1.to(device), img2.to(device)

            # İleri geçiş
            outputs = model(img1, img2)

            # Etiketler
            matches = labels["matches"]
            # convert non 1024 sized matches to 1024 size
            matches_converted = torch.ones((1024)) * -1
            matches_converted[0:len(matches[0])] = matches


            # Kayıp hesaplama
            loss = criterion(outputs, torch.tensor(matches_converted, dtype=torch.float32, device=device))
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
    return val_loss/len(val_loader)

# Veri kümesi
train_loader, val_loader, test_loader = get_dataloaders(root_dir="data/", batch_size=1, num_workers=0, train_split=0.8, val_split=0.1, seed=5)

# Model, loss ve optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RCM().to(device)
criterion = nn.MSELoss()  # Örneğin, kayıp fonksiyonu olarak Mean Squared Error kullanabilirsiniz
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Modeli eğit
train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)

print("Model eğitimi tamamlandı ve kaydedildi.")
