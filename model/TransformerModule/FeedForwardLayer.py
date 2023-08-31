import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out
    ):
        super().__init__()
        self.fc_in = nn.Linear(
            in_features=dim_in,
            out_features=dim_hidden
        )
        self.fc_out = nn.Linear(
            in_features=dim_hidden,
            out_features=dim_out
        )
        self.activation = nn.GELU()

    def forward(
        self,
        x
    ):
        x = self.fc_in(x)
        x = self.activation(x)
        x = self.fc_out(x)

        return x
