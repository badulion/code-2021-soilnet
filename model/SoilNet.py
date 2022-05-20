import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


class SoilNN(nn.Module):
    def __init__(self, input_size: int = 15, output_size: int = 3, hidden_size: int = 256, hidden_layers: int = 6, dropout=0.0):
        super().__init__()

        sizes = [input_size] + [hidden_size for i in range(hidden_layers)] + [output_size]
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=sizes[i], out_features=sizes[i+1], bias=True) for i in range(len(sizes)-1)])

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.SELU()
        self.normalizer = nn.Softmax(dim=1)
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='selu')

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        x = self.layers[-1](x)
        x = self.normalizer(x)

        return x


class SoilNNModule(pl.LightningModule):
    def __init__(self,
                 input_size,
                 output_size: int = 3,
                 hidden_size: int = 256,
                 hidden_layers: int = 6,
                 dropout=0.0,
                 learning_rate=0.001,
                 l2_regularization=0.0,
                 val_metric="val_loss"):
        super().__init__()

        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.loss = nn.functional.mse_loss
        self.val_metric = val_metric
        self.net = SoilNN(
            dropout=dropout,
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
        self.log(self.val_metric, loss, prog_bar=True)
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
                                                 patience=20, verbose=False, mode="min")
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
                 l2_regularization=0.0,
                 n_epochs=100):

        self.model = SoilNNModule(
             input_size=num_features,
             output_size=num_tasks,
             hidden_size=hidden_size,
             hidden_layers=hidden_layers,
             dropout=dropout,
             learning_rate=learning_rate,
             l2_regularization=l2_regularization,
        )

        self.checkpoint_callback_pretrain = ModelCheckpoint(monitor="val_loss")
        early_stop_callback_pretrain = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                            patience=20, verbose=False, mode="min")

        self.checkpoint_callback_finetune = ModelCheckpoint(monitor="val_loss")
        early_stop_callback_finetune = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                            patience=20, verbose=False, mode="min")
        self.trainer_pretrain = pl.Trainer(
             max_epochs=n_epochs,
             num_sanity_val_steps=0,
             val_check_interval=100,
             callbacks=[self.checkpoint_callback_pretrain, early_stop_callback_pretrain],
        )

        self.trainer_finetune = pl.Trainer(
             max_epochs=n_epochs,
             num_sanity_val_steps=0,
             val_check_interval=10,
             callbacks=[self.checkpoint_callback_finetune, early_stop_callback_finetune],
        )

    def fit(self, train_datamodule, pretrain_datamodule):
        self.trainer_pretrain.fit(self.model, pretrain_datamodule)
        self.model = SoilNNModule.load_from_checkpoint(self.checkpoint_callback_pretrain.best_model_path, learning_rate=0.0001)

        self.trainer_finetune.fit(self.model, train_datamodule)
        self.model = SoilNNModule.load_from_checkpoint(self.checkpoint_callback_finetune.best_model_path)

    def predict(self, datamodule):
        preds = self.trainer_finetune.predict(self.model, datamodule)
        return torch.vstack(preds).cpu().detach().numpy()
