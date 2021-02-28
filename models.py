
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

class CNN_VAE(nn.Module):
    def __init__(self,input_channels,image_size,latent_size,max_channels):
        super().__init__()

        
        self.image_size = image_size
        self.max_channels = max_channels
        
        self.latent_size = latent_size
        self.last_layer_dim = ((image_size // 8) ** 2) * max_channels
        
        self.learning_rate = 0.0001

        self.encoder = nn.Sequential(
                            self._conv(input_channels,max_channels // 4), #16x16x8
                            self._conv(max_channels // 4, max_channels // 2), #8x8x16
                            self._conv(max_channels // 2, max_channels), #4x4x32
        )

        self.latent_mean = self._linear(self.last_layer_dim,self.latent_size)
        self.latent_var = self._linear(self.last_layer_dim,self.latent_size)

        self.linear_upscale = self._linear(self.latent_size,self.last_layer_dim)
        
        
        self.decoder = nn.Sequential(
                            self._deconv(max_channels,max_channels // 2), #8x8x16
                            self._deconv(max_channels // 2, max_channels // 4), #16x16x8
                            self._deconv(max_channels // 4, input_channels), #32x32x3
                            nn.Sigmoid(),
        
        )
        
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,

        )
    # Forward functions
    def distribute(self,encoding):
        flatten = encoding.view(-1,self.last_layer_dim)
        return self.latent_mean(flatten),self.latent_var(flatten)
    

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        encoding = self.encoder(x)
        mean,variance = self.distribute(encoding)
        z = self.reparameterise(mean, variance)
        z_upscaled = self.linear_upscale(z)
        projection_unflatten = z_upscaled.view(-1,self.max_channels,4,4)
        return self.decoder(projection_unflatten), mean, variance
    
    # Backward functions
    @staticmethod
    def loss_function(prediction,label,mean,variance):
        MSE = nn.MSELoss(size_average=False)(prediction, label) / label.size(0)
        KL = ((mean**2 + variance.exp() - 1 - variance) / 2).mean()

        return MSE+KL
    
    def backward(self,prediction,label,mean,variance):
        
        loss = self.loss_function(prediction,label,mean,variance)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    # Layers
    def _conv(self,channel_size,kernel_num):
        return nn.Sequential(
            nn.Conv2d(channel_size,
                      kernel_num,
                      kernel_size = 4,
                      stride=2,
                      padding=1),
            #nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )
    
    
    def _deconv(self,channel_num,kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(channel_num,
                               kernel_num,
                               kernel_size=4,
                               stride = 2,
                               padding = 1),
            #nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )
    def _linear(self, in_size, out_size, relu = False):
        return nn.Sequential(
                    nn.Linear(in_size,out_size),
                    nn.ReLU(),
        ) if relu else nn.Linear(in_size,out_size)