"""
This file defines a convolutional encoder and decoder model for audio corruption and restoration.
To replace this model, change the `!new:` tag in the hyperparameter file
to refer to a built-in SpeechBrain model or another file containing
a custom PyTorch module.
Authors
 * Ada Tur 2023
 * Mirco Ravanelli 2023
 * Cem Subakan 2023
"""
import torch
import speechbrain as sb
from speechbrain.processing.features import STFT, ISTFT, spectral_magnitude
from torchvision import models
import torch.nn.functional as F

#Encoder model definition
class Encoder(torch.nn.Module):
    """This model defines a convolutional encoder model
        Arguments
        ---------

        Example
        -------
        >>> en_coder = Encoder()
            >>> input_feats = torch.rand([16, 1, 100, 40])
            >>> outputs = en_coder((input_feats, 10))
            >>> print(outputs.shape)
        torch.Size([16, 16, 94, 34])
        """
    def __init__(
            self,
    ):
        #Calling the super constructor, and then defining the encoder
        super().__init__()
        self.encoder_cnn = torch.nn.Sequential(

        torch.nn.Conv2d(1, 4, 3, stride=1, padding=0),
        torch.nn.ReLU(True),
        torch.nn.Conv2d(4, 8, 3, stride=1, padding=0),
        torch.nn.ReLU(True),
        torch.nn.Conv2d(8, 16, 3, stride=1, padding=0),
        #torch.nn.ReLU(True)

       )

    def forward(self, x):
        data, length = x
        x = self.encoder_cnn(data)
        return x

class Decoder(sb.nnet.containers.Sequential):
    """This model defines a convolutional decoder model
        Arguments
        ---------

        Example
        -------
        >>> input_feats = torch.rand([16, 16, 94, 34])
            >>> de_coder = Decoder(input_feats.shape)
            >>> outputs = de_coder((input_feats))
            >>> print(outputs.shape)
            torch.Size([16, 1, 100, 40])
        """
    def __init__(
            self,
            input_shape,
            activation=torch.nn.LeakyReLU,
    ):
        super().__init__(input_shape=input_shape)

        self.decoder_conv = torch.nn.Sequential(

           torch.nn.ConvTranspose2d(16, 8, 3, stride=1, output_padding=0),
           torch.nn.ReLU(True),
           torch.nn.ConvTranspose2d(8, 4, 3, stride=1, padding=0, output_padding=0),
           torch.nn.ReLU(True),
           torch.nn.ConvTranspose2d(4, 1, 3, stride=1, padding=0, output_padding=0)

       )

    def forward(self, x):
      x = self.decoder_conv(x)
      return x
