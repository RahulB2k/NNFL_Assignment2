import torch
import torch.nn as nn

from ConvS2S import ConvEncoder
from Attention import MultiHeadAttention, PositionFeedforward

class Encoder(nn.Module): # 1 Mark
    def __init__(self, conv_layers, hidden_dim, feed_forward_dim=2048):
        super(Encoder, self).__init__()
        self.conv=ConvEncoder(hidden_dim,conv_layers)
        self.attention=MultiHeadAttention(hidden_dim,n_heads=16)
        self.feed_forward=PositionFeedforward(hidden_dim,feed_forward_dim)

    def forward(self, input):
        """
        Forward Pass of the Encoder Class
        :param input: Input Tensor for the forward pass. 
        """
        result=input
        conv_result=self.conv.forward(result)
        mha_result= self.attention.forward(conv_result, conv_result, conv_result)
        final_result=self.feed_forward.forward(mha_result)
        return final_result
        
