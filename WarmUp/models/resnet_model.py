"""
This file loads in a ResNet model and defines a Classifier to use for digit recognition.
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


class ResNet(torch.nn.Module):
    """This model extracts X-vectors for speaker recognition
        Arguments
        ---------

        Example
        -------
        >>> compute_resnet = ResNet()
        >>> input_feats = torch.rand([1, 3, 224, 224])
        >>> outputs = compute_resnet((input_feats, 10))
        >>> print(outputs.shape)
        torch.Size([1, 1000])
        """
    def __init__(
            self,
    ):
        #Calling the super constructor, and then loading in ResNet
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)

        #Forward done with ResNet's eval method
    def forward(self, x):
        data, length = x
        self.resnet.eval()
        return self.resnet(data)

class Classifier(sb.nnet.containers.Sequential):
    """This class implements the last MLP on the top of ResNet features.
        Arguments
        ---------
        input_shape : tuple
            Expected shape of an example input.
        activation : torch class
            A class for constructing the activation layers.
        lin_blocks : int
            Number of linear layers.
        lin_neurons : int
            Number of neurons in linear layers.
        out_neurons : int
            Number of output neurons.
        Example
        -------
        >>> input_feats = torch.rand([16, 3, 224, 224])
        >>> compute_resnet = ResNet()
        >>> resnet = compute_resnet((input_feats, 10))
        >>> classify = Classifier(input_shape=resnet.shape)
        >>> output = classify(resnet)
        >>> output.shape
        torch.Size([16, 1211])
        """
    def __init__(
            self,
            input_shape,
            activation=torch.nn.LeakyReLU,
            lin_blocks=1,
            lin_neurons=512,
            out_neurons=1211,
    ):
        super().__init__(input_shape=input_shape)

        self.append(activation(), layer_name="act")
        self.append(sb.nnet.normalization.BatchNorm1d, layer_name="norm")

        if lin_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="DNN")

        # Adding fully-connected layers
        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(
                sb.nnet.containers.Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                sb.nnet.linear.Linear,
                n_neurons=lin_neurons,
                bias=True,
                layer_name="linear",
            )
            self.DNN[block_name].append(activation(), layer_name="act")
            self.DNN[block_name].append(
                sb.nnet.normalization.BatchNorm1d, layer_name="norm"
            )

        # Final Softmax classifier
        self.append(
            sb.nnet.linear.Linear, n_neurons=out_neurons, layer_name="out"
        )
        self.append(
            sb.nnet.activations.Softmax(apply_log=True), layer_name="softmax"
        )