import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        """
        The convolutional Encoder layer.
        Model based on the paper "Convolutional Sequence to Sequence Learning - https://arxiv.org/pdf/1705.03122.pdf"
        :param input_dim: input dimension of the tensor
        :param num_layers: Number of convolutional you desire to stack on top of each other
        """
        super(ConvEncoder, self).__init__()
        self.convolutions = self.get_convolutions(input_dim=input_dim, num_layers=num_layers)
    
    def forward(self, source): # 1 Mark
        """
        Remember, we have multiple convolutional layers, for iterate over them.
        :param source: Input tensor for the forward pass. 
        """
        for conv in self.convolutions:
            out=conv(source)
            out = nn.functional.glu(out, dim=1)
            source=out
        return source
    
    def get_convolutions(self, input_dim, num_layers=3): # 0.5 Marks
        """
        :param input_dim: input dimension
        :param num_layers: Number of convolutional you desire to stack on top of each other
        :return: nn.ModuleList()
        """
        module_list=nn.ModuleList()
        for _ in range(0, num_layers):
            module_list.append(nn.Conv1d(in_channels=input_dim, out_channels=2*input_dim, kernel_size=3, stride=1, padding=1))
        return module_list

        
        
