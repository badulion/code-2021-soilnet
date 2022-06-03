
# soilNet orientierung
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


class CNNModule(nn.Module):
    def __init__(self, input_channels: int = 16, output_size: int = 3, dropout=0.5, patch_size=5, base_channels=16):
        super().__init__()
        n = patch_size * 2 + 1
        self.layers = []
        self.layers.append(nn.Conv2d(input_channels, base_channels, kernel_size=(1,1))) # n x n    n = 11
        self.layers.append(torch.nn.ReLU())
        self.layers.append(nn.Conv2d(base_channels, base_channels * 2, kernel_size=(3,3))) # n-2 x n-2    n = 9
        self.layers.append(torch.nn.ReLU())
        self.layers.append(nn.MaxPool2d(2, 2)) # (n-2)/2 x (n-2)/2         n = 5
        self.layers.append(torch.nn.Dropout(p=dropout, inplace=False))
        self.layers.append(nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=(3,3))) # (n-2)/2-2 x (n-2)/2-2    n = 3
        self.layers.append(torch.nn.ReLU())
        self.layers.append(nn.Conv2d(base_channels * 2, base_channels, kernel_size=(1,1)))
        self.layers.append(nn.Flatten(1))
        inn = int(int((n-2)/2)-2) * int(int((n-2)/2)-2) * base_channels
        out = base_channels
        self.layers.append(nn.Linear(inn, out)) #methode mit der man inputsize erkennen kann
        self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Dropout(p=dropout, inplace=False))
        self.layers.append(nn.Linear(base_channels, base_channels//2))
        self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Dropout(p=dropout, inplace=False))
        self.layers.append(nn.Linear(base_channels//2, output_size))
        self.layers.append(torch.nn.ReLU())
        self.layers.append(nn.Softmax())

        # model sequential
        self.model = nn.Sequential(*self.layers)
        #in 6 outputs splitten am ende nach tiefe (ganz spät)
        # wie bestimme ich die sizes im neuronalen Netz? (in-channel, out-channel, kernel-size)
        # wie integriere ich patchDS und cnn am besten?
        # von 12 auf 16 dann kopieren

        #for layer in self.layers:
        #   nn.init.kaiming_normal_(layer.weight, nonlinearity='selu')

    def forward(self, x):
        return self.model(x)

# gleiche parameter wie für SoilCNN
class SoilCNNLightningModule(pl.LightningModule):
    def __init__(self,
                 input_channels: int = 16, 
                 output_size: int = 3, 
                 dropout: float = 0.5, 
                 patch_size: int = 5, 
                 base_channels: float = 16,
                 learning_rate=0.001,
                 l2_regularization=0.0):
        super().__init__()

        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.loss = nn.functional.mse_loss
        self.net = CNNModule(
            input_channels = input_channels, 
            output_size = output_size, 
            dropout = dropout, 
            patch_size = patch_size, 
            base_channels = base_channels,
        )
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        #get data
        x, y = batch
        pred = self.forward(x)

        #calculate loss
        loss = self.loss(pred, y)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)

        return loss

    def predict_step(self, batch, batch_idx):
        #get data
        x, y = batch
        pred = self.forward(x)
        return pred

    def validation_step(self, batch, batch_idx):
        #get data
        x, y = batch
        pred = self.forward(x)

        #calculate loss
        loss = self.loss(pred, y)

        # Logging to TensorBoard by default
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        #get data
        x, y = batch
        pred = self.forward(x)

        #calculate loss
        loss = self.loss(pred, y)

        # Logging to TensorBoard by default
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_regularization)


class SoilCNN:
    def __init__(self,
                 num_features,
                 output_size: int = 3, 
                 dropout: float = 0.5, 
                 patch_size: int = 5, 
                 base_channels: float = 16,
                 learning_rate=0.001,
                 l2_regularization=0.0,
                 n_epochs=100):

        #save parameters
        self.num_features = num_features

        self.model = SoilCNNLightningModule(
                input_channels = num_features,
                output_size = output_size,
                patch_size = patch_size, 
                base_channels = base_channels,
                dropout=dropout,
                learning_rate=learning_rate,
                l2_regularization=l2_regularization,
        )
        self.checkpoint_callback = ModelCheckpoint(monitor="val_loss")
        self.early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                                 patience=5, verbose=False, mode="min")
        self.trainer = pl.Trainer(
             max_epochs=n_epochs,
             num_sanity_val_steps=0,
             callbacks=[self.early_stop_callback, self.checkpoint_callback],
        )

    def fit(self, datamodule):
        self.trainer.fit(self.model, datamodule)
        self.model = SoilCNNLightningModule.load_from_checkpoint(self.checkpoint_callback.best_model_path)

    def predict(self, datamodule):
        preds = self.trainer.predict(self.model, datamodule)
        return torch.vstack(preds).cpu().detach().numpy()