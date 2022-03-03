import gpytorch
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, kernel, num_features,
                 num_latents, num_tasks,
                 num_inducing_points=64):
        # parameters
        self.num_tasks = num_tasks
        self.num_latents = num_latents
        self.num_features = num_features

        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_latents, 128, num_features)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = self._create_kernel(kernel.name, kernel.parameters)

    def _create_kernel(self, kernel, kernel_parameters={}):
        if kernel == "rbf":
            base_kernel = gpytorch.kernels.RFFKernel(
                ard_num_dims=self.num_features,
                batch_shape=torch.Size([self.num_latents]),
                **kernel_parameters)
        elif kernel == "spectralmixture":
            base_kernel = gpytorch.kernels.SpectralMixtureKernel(
                ard_num_dims=self.num_features,
                batch_shape=torch.Size([self.num_latents]),
                **kernel_parameters)
        else:
            base_kernel = gpytorch.kernels.MaternKernel(
                ard_num_dims=self.num_features,
                batch_shape=torch.Size([self.num_latents]),
                **kernel_parameters)

        full_kernel = gpytorch.kernels.ScaleKernel(
            base_kernel,
            batch_shape=torch.Size([self.num_latents])
        )
        return full_kernel

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class IndependentGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, kernel, num_features,
                 num_latents, num_tasks,
                 num_inducing_points=64):
        # parameters
        self.num_tasks = num_tasks
        self.num_latents = num_latents
        self.num_features = num_features

        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(num_tasks, num_inducing_points, num_features)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = self._create_kernel(kernel.name, kernel.parameters)

    def _create_kernel(self, kernel, kernel_parameters={}):
        if kernel == "rbf":
            base_kernel = gpytorch.kernels.RFFKernel(
                ard_num_dims=self.num_features,
                batch_shape=torch.Size([self.num_tasks]),
                **kernel_parameters)
            full_kernel = gpytorch.kernels.ScaleKernel(
                base_kernel,
                batch_shape=torch.Size([self.num_tasks])
            )
        elif kernel == "spectralmixture":
            full_kernel = gpytorch.kernels.SpectralMixtureKernel(
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

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGPLightningModule(pl.LightningModule):
    def __init__(self, num_features, kernel, num_data, num_tasks=3, num_latents=2,
                 num_inducing_points=64, learning_rate=0.1, n_epochs=10, val_metric='mse'):
        #warnings.filterwarnings("ignore", category=UserWarning)  # GPytorch bug triggers warnings
        super().__init__()
        self.save_hyperparameters()
        self.num_tasks = num_tasks
        self.learning_rate = learning_rate
        self.val_metric = val_metric
        self.n_epochs = n_epochs
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        self.model = IndependentGPModel(num_latents=num_latents, num_tasks=num_tasks, num_features=num_features,
                                        num_inducing_points=num_inducing_points, kernel=kernel)
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=num_data)

    def training_step(self, batch, batch_idx):
        #get data
        x, y = batch
        output = self.model(x)

        #calculate loss
        loss = -self.mll(output, y)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)

        return loss

    def predict_step(self, batch, batch_idx):
        self.model.eval()
        self.likelihood.eval()
        #get data
        x, y = batch
        output = self.likelihood(self.model(x))
        return output.mean

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        self.likelihood.eval()
        #get data
        x, y = batch
        output = self.model(x)

        #calculate loss
        if self.val_metric == "mll":
            loss = -self.mll(output, y)
        else:
            pred = self.likelihood(output).mean
            loss = torch.nn.functional.mse_loss(pred, y)

        # Logging to TensorBoard by default
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        #get data
        x, y = batch
        output = self.model(x)

        #calculate loss
        loss = -self.mll(output, y)

        # Logging to TensorBoard by default
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self.learning_rate)
        return optimizer


class VariationalGP:
    def __init__(self,
                 num_features,
                 kernel,
                 num_data,
                 num_tasks: int = 3,
                 num_latents: int = 6,
                 num_inducing_points=64,
                 learning_rate=0.001,
                 val_metric="mae",
                 n_epochs=100):

        #save parameters
        self.num_features = num_features

        self.model = VariationalGPLightningModule(
            num_features,
            kernel,
            num_data=num_data,
            num_tasks=num_tasks,
            num_latents=num_latents,
            num_inducing_points=num_inducing_points,
            learning_rate=learning_rate,
            val_metric=val_metric,
            n_epochs=n_epochs
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
        self.model = VariationalGPLightningModule.load_from_checkpoint(self.checkpoint_callback.best_model_path)

    def predict(self, datamodule):
        preds = self.trainer.predict(self.model, datamodule)
        return torch.vstack(preds).cpu().detach().numpy()
