
import torch
from torch import nn

class VAE(nn.Module):
	def __init__(self,learning_rate = 1e-3,latent_dimension=20):
		super().__init__()

		self.d = latent_dimension
		self.learning_rate = learning_rate


		self.encoder = nn.Sequential(
			nn.Linear(784, self.d ** 2),
			nn.ReLU(),
			nn.Linear(self.d ** 2, self.d * 2)
		)

		self.decoder = nn.Sequential(
			nn.Linear(self.d, self.d ** 2),
			nn.ReLU(),
			nn.Linear(self.d ** 2, 784),
			nn.Sigmoid(),
		)

		self.optimizer = torch.optim.Adam(
			self.parameters(),
			lr=learning_rate,
		)

	#Forward functions
	def reparameterise(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = std.data.new(std.size()).normal_()
			return eps.mul(std).add_(mu)
		else:
			return mu

	def forward(self, x):
		mu_logvar = self.encoder(x.view(-1, 784)).view(-1, 2, self.d)
		mu = mu_logvar[:, 0, :]
		logvar = mu_logvar[:, 1, :]
		z = self.reparameterise(mu, logvar)
		return self.decoder(z), mu, logvar

	# Backward functions
	@staticmethod    
	def loss_function(prediction,actual,mean,variance):
		BCE = nn.functional.binary_cross_entropy(
		prediction, actual.view(-1, 784), reduction='sum')

		KLD = 0.5 * torch.sum(variance.exp() - variance - 1 + mean.pow(2))

		return BCE + KLD

	def backward(self,prediction,actual,mean,variance):

			loss = self.loss_function(prediction,actual,mean,variance)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			return loss.item()