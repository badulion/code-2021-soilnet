import gpytorch
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 num_features,
                 num_tasks,
                 kernel,
                 feature_extractor=nn.Identity()):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = self._create_kernel(kernel.name, kernel.parameters)

    def forward(self, x):
        x = self.feature_extractor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

    def _create_kernel(self, kernel, kernel_parameters={}):
        if kernel == "rbf":
            base_kernel = gpytorch.kernels.RFFKernel(
                ard_num_dims=self.num_features,
                batch_shape=torch.Size([self.num_tasks]),
                **kernel_parameters)
        elif kernel == "spectralmixture":
            base_kernel = gpytorch.kernels.SpectralMixtureKernel(
                ard_num_dims=self.num_features,
                batch_shape=torch.Size([self.num_tasks]),
                **kernel_parameters)
        else:
            base_kernel = gpytorch.kernels.MaternKernel(
                ard_num_dims=self.num_features,
                batch_shape=torch.Size([self.num_tasks]),
                **kernel_parameters)

        full_kernel = gpytorch.kernels.ScaleKernel(
            base_kernel,
            batch_shape=torch.Size([self.num_tasks])
        )
        return full_kernel


class ExactGPModel:
    def __init__(self, num_features, kernel,
                 iterations=100, learning_rate=0.01, num_tasks=3):
        self.num_tasks = num_tasks
        self.num_features = num_features
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        self.model = None
        self.kernel = kernel
        self.iterations = iterations
        self.learning_rate = learning_rate

    def fit(self, X, y, iterations=90):
        X = torch.tensor(X, dtype=torch.float32).contiguous()
        y = torch.tensor(y, dtype=torch.float32).contiguous()

        feature_extractor = nn.Identity()
        self.model = MultitaskGPModel(X, y, self.likelihood,
                                      num_features=self.num_features,
                                      num_tasks=self.num_tasks,
                                      kernel=self.kernel,
                                      feature_extractor=feature_extractor)
        loss = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # setting model to train mode
        self.model.train()
        self.likelihood.train()

        for i in range(iterations):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(X)
            # Calc loss and backprop gradients
            output = -loss(output, y)
            output.backward()
            optimizer.step()
            print(f'Iter {i}/{iterations} - Loss: {output.item()}')

    def predict(self, X, y=None):
        self.model.eval()
        self.likelihood.eval()
        X = torch.tensor(X, dtype=torch.float32).contiguous()
        predictions = self.likelihood(self.model(X))
        mean = predictions.mean
        return mean.detach().numpy()

    def predict_variance(self, X):
        self.model.eval()
        self.likelihood.eval()
        X = torch.tensor(X, dtype=torch.float32).contiguous()
        predictions = self.likelihood(self.model(X))
        return predictions._covar.diag().detach().numpy()

    def pretrain_extractor(self, X, y):
        dataset = GenericDataModule(X, y)
        autoencoder = AutoEncoder(231)
        trainer = pl.Trainer(max_epochs=20)
        trainer.fit(autoencoder, dataset)
        return autoencoder.encoder


class AutoEncoder(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim: int = 64, latent_dim: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                     nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.SiLU(),
                                     nn.Linear(hidden_dim, latent_dim),
                                     nn.Tanh())

        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                     nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.SiLU(),
                                     nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x, x_hat)
        self.log(f"loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class GenericDataModule(pl.LightningDataModule):
    def __init__(self, X, y, batch_size: int = 32):
        super().__init__()
        self.train = GenericDataset(X, y)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)


class GenericDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, i: int) -> tuple:
        x = self.X[i, :]
        y = self.y[i, :]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

    def __len__(self) -> int:
        return len(self.X)
