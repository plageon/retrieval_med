import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, AutoModel


class MyRobertaModel(PreTrainedModel):
    def __init__(self, config, num_T_period, num_N_period, num_cancer_type, model_name_or_path):
        super().__init__(config)
        self.roberta = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.T_period_linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, num_T_period),
        )
        self.N_period_linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, num_N_period),
        )
        self.cancer_type_linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, num_cancer_type),
        )
        self.loss_func = CrossEntropyLoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            # T_period_labels=None,
            # N_period_labels=None,
            # cancer_type_labels=None,
    ):
        T_period_labels, N_period_labels, cancer_type_labels = labels
        batch_size = input_ids.size(0)
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        cls_outputs = last_hidden_state[:, 0]
        T_period_logits = self.T_period_linear(cls_outputs)
        N_period_logits = self.N_period_linear(cls_outputs)
        cancer_type_logits = self.cancer_type_linear(cls_outputs)
        loss = None
        if T_period_labels is not None:
            T_period_loss = self.loss_func(T_period_logits, T_period_labels)
            N_period_loss = self.loss_func(N_period_logits, N_period_labels)
            cancer_type_loss = self.loss_func(cancer_type_logits, cancer_type_labels)
            loss = T_period_loss + N_period_loss + cancer_type_loss

        return {
            'loss': loss,
            # 'T_period_logits': T_period_logits,
            # 'N_period_logits': N_period_logits,
            # 'cancer_type_logits': cancer_type_logits,
            'logits': [T_period_logits, N_period_logits, cancer_type_logits],
        }
