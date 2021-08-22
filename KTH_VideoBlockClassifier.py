import torch 
import torch.nn as nn 
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.nn import functional as F

class KTH_VideoBlockClassifier(pl.LightningModule):   
    def __init__(self):
        super(KTH_VideoBlockClassifier, self).__init__()
                                                            # input video block shape (1,15,120,160) - with N (batch)
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(4, 5, 5)),        # output Size (16,12,116,156)
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),            # output Size (16,12,58,78) 
            nn.Dropout(0.5))

        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(4, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),            # output Size (32,4,28,38) 
            nn.Dropout(0.5))

        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),            # output Size (64,1,13,18) 
            nn.Dropout(0.5))

        self.fc1 = nn.Linear(64 * 1 * 13 * 18, 128)
        self.dropfc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out = self.dropfc1(out)
        out = self.fc2(out)

        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)
 
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        y_pred = torch.argmax(logits, dim=1)
        accuracy = torch.sum(y == y_pred).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output