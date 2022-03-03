from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern


class SKLearnGPModel:
    def __init__(self, nu=2.5):
        self.kernel = WhiteKernel()+Matern(nu=nu)
        self.gp_regressor = GaussianProcessRegressor(kernel=self.kernel)

    def fit(self, X, y):
        self.gp_regressor.fit(X, y)

    def predict(self, X, y=None):
        return self.gp_regressor.predict(X)
