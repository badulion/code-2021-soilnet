import torch
import torch.nn as nn
from torchviz.dot import make_dot
from graphviz import Digraph
import torch
from torch.autograd import Variable, Function


class WeightedIDW(nn.Module):
    def __init__(self, train_x, train_y, num_features, n_neighbors=5, p=2, out_of_k=0.0):
        super(WeightedIDW, self).__init__()
        self.num_features = num_features
        self.n_neighbors = n_neighbors
        self.p = p
        self.out_of_k = out_of_k
        self.weights = nn.Parameter(torch.zeros(num_features, dtype=torch.float32))
        self.train_x = train_x
        self.train_y = train_y
        self.mask = self.location_mask(train_x)
        assert self.out_of_k <= 1 and self.out_of_k >= 0

    def forward(self, x):
        if self.training == True:
            assert torch.sum(torch.abs(x - self.train_x)) < 1e-6
            x = self.train_x / torch.exp(self.weights)
            weights = self.neighbor_weights(x, x)
        else:
            x = x / torch.exp(self.weights)
            x_train = self.train_x / torch.exp(self.weights)
            weights = self.neighbor_weights(x, x_train)

        return torch.mm(weights, self.train_y)

    def neighbor_weights(self, x, y):
        dist = 1/self.distance_matrix(x, y)
        # andere Beispiele aus der selben location auch auf 0 setzten
        # mask berechnen -> welche werte dürfen nicht berücksichtigt werden
        if self.training:
            #dist.fill_diagonal_(0)
            dist.masked_fill_(self.mask, 0)
        value, _ = torch.topk(dist, k=self.n_neighbors)
        min_val, _ = torch.min(value, dim=1, keepdim=True)
        if self.training:
            weights = (dist >= min_val.expand(dist.shape))*dist + self.out_of_k*dist*(dist < min_val.expand(dist.shape))
        else:
            weights = (dist >= min_val.expand(dist.shape))*dist
        # weights gewichten?
        weights = weights/torch.sum(weights, dim=1, keepdim=True)
        return weights

    def distance_matrix(self, x, y=None):
        # adapted from https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3
        y = x if y is None else y

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        # distanz hoch 'w' als zusätzlicher parameter
        dist = (torch.pow(x - y, self.p).sum(2)+1e-6) ** (1.0/self.p)

        return dist

    def location_mask(self, x, location_features=[0, 1]):

        x = x[:, location_features]
        n = x.size(0)
        d = x.size(1)

        x1 = x.unsqueeze(1).expand(n, n, d)
        x2 = x.unsqueeze(0).expand(n, n, d)

        # distanz hoch 'w' als zusätzlicher parameter
        mask = torch.abs(x1 - x2).sum(2)

        return mask < 1e-6


class WeightedIDWModel:
    def __init__(self, num_features, n_neighbors=5, p=2, out_of_k=0.0, lr=0.1, n_iterations=10):
        self.num_features = num_features
        self.n_neighbors = n_neighbors
        self.n_iterations = n_iterations
        self.out_of_k = out_of_k
        self.p = p
        self.model = None
        self.lr = lr

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        self.model = WeightedIDW(X, y, self.num_features, n_neighbors=self.n_neighbors,
                                 p=self.p, out_of_k=self.out_of_k)
        self.model.train()

        # training loop to learn the feature weights
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        loss = nn.functional.mse_loss
        for i in range(self.n_iterations):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            pred = self.model(X)

            # Calc loss and backprop gradients
            output = loss(pred, y)
            output.backward()
            optimizer.step()
            lr_scheduler.step(output)
            #print(f'Iter {i}/{self.n_iterations} - Loss: {output.item()}')

    def predict(self, X, y=None):
        assert self.model is not None
        self.model.eval()

        X = torch.tensor(X, dtype=torch.float32)
        pred = self.model(X)
        return pred.detach().numpy()
