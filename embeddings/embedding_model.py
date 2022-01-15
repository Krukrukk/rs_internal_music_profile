from re import I
import torch
from torch import nn


class EmbeddingModel(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        self.input_features_dim = kwargs.get("input_features_dim", 11)
        self.input_text_dim = kwargs.get("input_text_dim", 2304)
        self.hidden_text_dim = kwargs.get("hidden_text_dim", 16)
        self.input_lstm_dim = kwargs.get("input_lstm_dim", 1)
        self.hidden_lstm_dim = kwargs.get("hidden_lstm_dim", 16)
        self.num_layers_lstm = kwargs.get("num_layers_lstm", 2)
        self.hidden_dense_dim = kwargs.get("num_layers_lstm", 32)
        self.output_dim = kwargs.get("output_dim", 16)
        self.lstmSections = nn.LSTM(
            input_size=self.input_lstm_dim,
            hidden_size=self.hidden_lstm_dim,
            num_layers=self.num_layers_lstm,
            batch_first=True,
        )  # lstm
        self.fc_text = nn.Linear(self.input_text_dim, 16)
        merge_size = (
            self.input_features_dim + self.hidden_text_dim + self.hidden_lstm_dim
        )
        self.fc1 = nn.Linear(merge_size, self.hidden_dense_dim)
        self.fc2 = nn.Linear(self.hidden_dense_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(
        self,
        features: torch.Tensor,
        embedds: torch.Tensor,
        sections: torch.Tensor,
    ) -> torch.Tensor:
        # Output Text
        embedds = self.relu(self.fc_text(embedds.clone().detach().requires_grad_(True)))

        # Output LSTM
        outputSections, (hnSections, cnSections) = self.lstmSections(
            sections.clone().detach().requires_grad_(True)
        )
        hnSections = hnSections[-1].view(-1, self.hidden_lstm_dim)

        # Output cat
        output = self.relu(
            self.fc1(
                torch.cat(
                    (
                        embedds,
                        hnSections,
                        features.clone().detach().requires_grad_(True),
                    ),
                    axis=1,
                ).float()
            )
        )
        output = self.fc2(output)
        # Now process together
        output = nn.functional.normalize(output, 2)
        return output
