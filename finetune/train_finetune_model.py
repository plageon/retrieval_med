import os

import torch
from transformers import HfArgumentParser, TrainingArguments, AutoConfig, AutoTokenizer
import numpy as np

from period_type.argparser import ModelArguments, DataTrainingArguments
from period_type.model import MyRobertaModel
from period_type.collect_data import LabelClusters, load_datasets, collator_fn
from period_type.trainer import MyTrainer
from period_type.util import compute_metrics


class PeriodTypeModel:
    def __init__(self, ):
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_type = 'roberta'
        self.k_shot = 1
        self.model_name = 'chinese-{}-wwm-ext'.format(model_type)
        self.model_args, self.data_args, self.training_args = parser.parse_json_file(
            "finetune/{}_finetune_args.json".format(model_type))
        if self.k_shot > 0:
            assert str(self.k_shot) in self.data_args.train_file, "incorrespondent train file, {} shot, {}".format(
                self.k_shot, self.data_args.train_file)
        device = torch.device("cuda")
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        self.compute_metrics = compute_metrics
        label_cluster = LabelClusters()
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)

        self.model = MyRobertaModel(config, len(label_cluster.T_periods2id), len(label_cluster.N_periods2id),
                                    len(label_cluster.cancer_types2id),
                                    model_name_or_path=self.model_args.model_name_or_path)

        # self.model.load_state_dict(
        #     torch.load('output/finetune/{}/pytorch_model.bin'.format(self.model_name), map_location=device))
        # self.model.from_pretrained(self.training_args.output_dir)

    def train_period_type(self):
        train_dataset, eval_dataset, test_dataset = load_datasets(self.data_args, self.roberta_tokenizer)
        trainer = MyTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.roberta_tokenizer,
            data_collator=collator_fn,
            not_bert_learning_rate=self.model_args.not_bert_learning_rate,
        )
        # Training
        last_checkpoint = None
        if self.training_args.do_train:
            checkpoint = None
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
            elif os.path.isdir(self.model_args.model_name_or_path):
                # Check the config from that potential checkpoint has the right number of labels before using it as a
                # checkpoint.
                checkpoint = self.model_args.model_name_or_path

            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)

            trainer.save_model()  # Saves the tokenizer too for easy upload

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        if self.training_args.do_eval:
            # on dev
            p = trainer.predict(test_dataset=eval_dataset)
            eval_result = compute_metrics(p)

            # on test
            p = trainer.predict(test_dataset=test_dataset)
            test_result = compute_metrics(p)

            if self.k_shot > 0:
                res_file = 'output/finetune/results/{}_{}_shot_res'.format(self.model_name, self.k_shot)
            else:
                res_file = 'output/finetune/results/{}_res'.format(self.model_name)
            with open(res_file, "w") as writer:
                print("***** Eval results *****")
                for key, value in eval_result.items():
                    print("eval  {} = {}".format(key, value))
                    writer.write("eval {} = {}\n".format(key, value))
                print("***** Test results *****")
                for key, value in test_result.items():
                    print("test  {} = {}".format(key, value))
                    writer.write("test {} = {}\n".format(key, value))

    def preprocess_function(self, examples):
        label_clusters = self.label_clusters

        VOCAB_PAD = 0
        LABEL_PAD = -100

        input_ids = []
        T_period_ids = []
        N_period_ids = []
        cancer_type_ids = []

        for example in examples:
            oritext = example["text"]

            tokens = self.roberta_tokenizer.tokenize(oritext)
            input_id = self.roberta_tokenizer.convert_tokens_to_ids(tokens)
            # T_period_id = label_clusters.T_periods2id[example['T_period']]
            # N_period_id = label_clusters.N_periods2id[example['N_period']]
            # cancer_type_id = label_clusters.cancer_types2id[example['cancer_type']]
            input_ids.append(torch.LongTensor(input_id))
            # T_period_ids.append(T_period_id)
            # N_period_ids.append(N_period_id)
            # cancer_type_ids.append(cancer_type_id)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=VOCAB_PAD)
        # T_period_ids = torch.LongTensor(T_period_ids)
        # N_period_ids = torch.LongTensor(N_period_ids)
        # cancer_type_ids = torch.LongTensor(cancer_type_ids)

        result = {
            'input_ids': input_ids,
            'attention_mask': input_ids != VOCAB_PAD,
            'labels': [None, None, None],
            # "cancer_type_labels": cancer_type_ids,
            # "T_period_labels": T_period_ids,
            # "N_period_labels": N_period_ids,

        }
        return result

    def do_period_type(self, texts):
        self.label_clusters = LabelClusters()
        inputs = self.preprocess_function(texts)
        if not inputs:
            return None
        self.model.eval()
        outputs = [o.detach().numpy() for o in self.model(**inputs)['logits']]
        pred_attribute_ids = [np.argmax(output, axis=1) for output in outputs]
        pred_T_periods = [self.label_clusters.id2T_periods[i] for i in pred_attribute_ids[0]]
        pred_N_periods = [self.label_clusters.id2N_periods[i] for i in pred_attribute_ids[1]]
        pred_cancer_types = [self.label_clusters.id2cancer_types[i] for i in pred_attribute_ids[2]]
        for pred_T_period, pred_N_period, pred_cancer_type, text in zip(pred_T_periods, pred_N_periods,
                                                                        pred_cancer_types,
                                                                        texts):
            text['T_period'] = pred_T_period
            text['N_period'] = pred_N_period
            text['cancer_type'] = pred_cancer_type

        # print(texts)
        return texts


if __name__ == '__main__':
    texts = [{'text': '1.(膀胱) 小块粘膜组织呈慢性炎性反应。'}]
    period_type_model = PeriodTypeModel()
    # res=period_type_model.do_period_type(texts)
    period_type_model.train_period_type()
    # print(res)
