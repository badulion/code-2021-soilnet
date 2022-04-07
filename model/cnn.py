
# soilNet orientierung
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


class SoilNN(nn.Module):
    def __init__(self, input_channels: int = 16, output_size: int = 3, dropout=0.5, n=5, base_channels=16):
        super().__init__()

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
        self.layers.append(nn.Linear(int((n-2)/2-1 * (n-2)/2-1 * base_channels), base_channels)) #methode mit der man inputsize erkennen kann
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


class SoilNNModule(pl.LightningModule):
    def __init__(self,
                 input_size,
                 output_size: int = 3,
                 hidden_size: int = 256,
                 hidden_layers: int = 6,
                 dropout=0.0,
                 learning_rate=0.001,
                 l2_regularization=0.0):
        super().__init__()

        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.loss = nn.functional.mse_loss
        self.net = SoilNN(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
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


class SoilMLP:
    def __init__(self,
                 num_features: int = 15,
                 num_tasks: int = 3,
                 hidden_size: int = 256,
                 hidden_layers: int = 6,
                 dropout=0.0,
                 learning_rate=0.001,
                 l2_regularization=0.0,
                 n_epochs=100):

        #save parameters
        self.num_features = num_features

        self.model = SoilNNModule(
             input_size=num_features,
             output_size=num_tasks,
             hidden_size=hidden_size,
             hidden_layers=hidden_layers,
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
        self.model = SoilNNModule.load_from_checkpoint(self.checkpoint_callback.best_model_path)

    def predict(self, datamodule):
        preds = self.trainer.predict(self.model, datamodule)
        return torch.vstack(preds).cpu().detach().numpy()


class SoilNet:
    def __init__(self,
                 num_features: int = 15,
                 num_tasks: int = 3,
                 hidden_size: int = 256,
                 hidden_layers: int = 6,
                 dropout=0.5,
                 learning_rate=0.001,
                 n_epochs=100):

        self.model = SoilNNModule(
             input_size=num_features,
             output_size=num_tasks,
             hidden_size=hidden_size,
             hidden_layers=hidden_layers,
             dropout=dropout,
             learning_rate=learning_rate,
        )
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                            patience=5, verbose=False, mode="min")
        self.trainer = pl.Trainer(
             max_epochs=n_epochs,
             num_sanity_val_steps=0,
             val_check_interval=10,
             callbacks=[early_stop_callback],
        )

    def fit(self, train_datamodule, pretrain_datamodule):
        self.trainer.fit(self.model, pretrain_datamodule)
        self.trainer.fit(self.model, train_datamodule)

    def predict(self, datamodule):
        preds = self.trainer.predict(self.model, datamodule)
        return torch.vstack(preds).cpu().detach().numpy()
