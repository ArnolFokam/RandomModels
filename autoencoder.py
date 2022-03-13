import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()        
        # N, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # -> N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7) # -> N, 64, 1, 1
        )
        
        # N , 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28  (N,1,27,27)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

def train():
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                            batch_size=64,
                                            shuffle=True)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=1e-3, 
                                weight_decay=1e-5)

    # Point to training loop video
    num_epochs = 10
    outputs = []
    for epoch in range(num_epochs):
        for (img, _) in data_loader:
            # img = img.reshape(-1, 28*28) # -> use for Autoencoder_Linear
            recon = model(img)
            loss = criterion(recon, img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        outputs.append((epoch, img, recon))

    for k in range(0, num_epochs, 4):
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i+1)
            # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
            # item: 1, 28, 28
            plt.imshow(item[0])
                
        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2, 9, 9+i+1) # row_length + i + 1
            # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
            # item: 1, 28, 28
            plt.imshow(item[0])

    plt.savefig("fig.png")

    print("plotting embeddings")
    outputs_emb = []
    outputs_label = []

    for (img, _) in data_loader:
        # img = img.reshape(-1, 28*28) # -> use for Autoencoder_Linear
        outputs_emb.append(model.encode(img).detach().numpy())
        outputs_label.append(_.detach().numpy())

    outputs_emb_ex = np.squeeze(outputs_emb[0])
    outputs_label_ex = outputs_label[0]

    scaled_emb = StandardScaler().fit_transform(outputs_emb_ex)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(scaled_emb)

    for g in np.unique(outputs_label_ex):
        ix = np.where(outputs_label_ex == g)
        plt.scatter(
            embedding[ix, 0],
            embedding[ix, 1],
            c=[sns.color_palette()[g]],
            label=g)
    plt.gca().set_aspect('equal', 'datalim')
    plt.legend()
    plt.title('UMAP projection of the mnist dataset', fontsize=24)