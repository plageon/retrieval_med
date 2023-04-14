import json
import torch
from datasets import load_dataset


class LabelClusters:

    def __init__(self):
        with open('./data/cancer_types.dict', 'r', encoding='utf-8') as f:
            self.cancer_types2id = json.loads(f.read())
        self.id2cancer_types = {v: k for k, v in self.cancer_types2id.items()}

        with open('./data/T_periods.dict', 'r', encoding='utf-8') as f:
            self.T_periods2id = json.loads(f.read())
        self.id2T_periods = {v: k for k, v in self.T_periods2id.items()}

        with open('./data/N_periods.dict', 'r', encoding='utf-8') as f:
            self.N_periods2id = json.loads(f.read())
        self.id2N_periods = {v: k for k, v in self.N_periods2id.items()}


def preprocess_function(example, idx, split, domain_attribute, tokenizer):
    oritext = example["text"]

    tokens = tokenizer.tokenize(oritext)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    T_period_id = domain_attribute.T_periods2id[example['T_period']]
    N_period_id = domain_attribute.N_periods2id[example['N_period']]
    cancer_type_id = domain_attribute.cancer_types2id[example['cancer_type']]

    result = {
        "doc_id": example["id"],
        "input_ids": input_ids,
        "cancer_type_id": cancer_type_id,
        "T_period_id": T_period_id,
        "N_period_id": N_period_id,

    }
    return result


def load_datasets(data_args, tokenizer):
    full_datasets = load_dataset("json", data_files={'train': data_args.train_file,
                                                     'validation': data_args.validation_file,
                                                     'test': data_args.test_file}, cache_dir="./cache/")
    full_datasets.cleanup_cache_files()

    domain_attribute = LabelClusters()

    preprocess_args = {'split': 'train', 'domain_attribute': domain_attribute, 'tokenizer': tokenizer}
    train_dataset = full_datasets["train"].map(preprocess_function, batched=False, with_indices=True,
                                               load_from_cache_file=not data_args.overwrite_cache,
                                               fn_kwargs=preprocess_args)
    eval_dataset = full_datasets["validation"].map(preprocess_function, batched=False, with_indices=True,
                                                   load_from_cache_file=not data_args.overwrite_cache,
                                                   fn_kwargs=preprocess_args)
    test_dataset = full_datasets["test"].map(preprocess_function, batched=False, with_indices=True,
                                             load_from_cache_file=not data_args.overwrite_cache,
                                             fn_kwargs=preprocess_args)

    return train_dataset, eval_dataset, test_dataset


def collator_fn(examples):
    VOCAB_PAD = 0
    LABEL_PAD = -100

    input_ids = []
    T_period_ids = []
    N_period_ids = []
    cancer_type_ids = []

    for example in examples:
        input_ids.append(torch.LongTensor(example['input_ids']))
        T_period_ids.append(example['T_period_id'])
        N_period_ids.append(example['N_period_id'])
        cancer_type_ids.append(example['cancer_type_id'])

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=VOCAB_PAD)
    T_period_ids = torch.LongTensor(T_period_ids)
    N_period_ids = torch.LongTensor(N_period_ids)
    cancer_type_ids = torch.LongTensor(cancer_type_ids)

    result = {
        'input_ids': input_ids,
        'attention_mask': input_ids != VOCAB_PAD,
        'labels': [T_period_ids, N_period_ids, cancer_type_ids],
        # "cancer_type_labels": cancer_type_ids,
        # "T_period_labels": T_period_ids,
        # "N_period_labels": N_period_ids,

    }
    return result
