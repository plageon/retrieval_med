import json
from sklearn.metrics import f1_score, accuracy_score

cancer_types = {
    "无": 0,
    "肌层浸润性膀胱癌": 1,
    "非肌层浸润性膀胱癌": 2,
    "无法确定": 3,
    "非尿路上皮癌": 4,
    "未知": 5
}

T_periods = {
    "无": 0,
    "T2+": 1,
    "T2": 2,
    "T1": 3,
    "T3前": 4,
    "无法确定": 5,
    "Ta": 6,
    "T2b": 7,
    "T4a": 8,
    "T2a": 9,
    "Tis": 10,
    "T1+": 11,
    "T3": 12,
    "未知": 13,
    "T3+": 14,
    "T3a": 15,
    "T4": 16
}

N_periods = {
    "无": 0,
    "无法确定": 1,
    "N0": 2,
    "N1": 3,
    "N2": 4,
    "N3": 5,
    "未知": 6,
    "Nx": 7
}

if __name__ == "__main__":
    splits = ['dev', 'test']
    top_ks = [5, 10]
    for split in splits:
        for top_k in top_ks:
            with open('output/retrieval/{}_top_{}_ans.jsonl'.format(split, top_k), 'r', encoding='utf-8') as f:
                answer_lines = f.readlines()

            gold_cancer_types, gold_T_periods, gold_N_periods = [], [], []
            pred_cancer_types, pred_T_periods, pred_N_periods = [], [], []
            eval_res = {}
            for line in answer_lines:
                line_data = json.loads(line)
                gold_cancer_type, gold_T_period, gold_N_period = line_data['cancer_type'], line_data['T_period'], \
                                                                 line_data[
                                                                     'N_period']
                pred_cancer_type, pred_T_period, pred_N_period = line_data['answer_cancer_type'], line_data[
                    'answer_T_period'], line_data['answer_N_period']
                gold_cancer_types.append(cancer_types[gold_cancer_type])
                gold_T_periods.append(T_periods[gold_T_period])
                gold_N_periods.append(N_periods[gold_N_period])

                # if any class name in answer else "无法确定"
                pred_cancer_types.append(([v for k, v in cancer_types.items() if k in pred_cancer_type] + [3])[0])
                pred_T_periods.append(([v for k, v in T_periods.items() if k in pred_T_period] + [5])[0])
                pred_N_periods.append(([v for k, v in N_periods.items() if k in pred_N_period] + [1])[0])

            eval_res['cancer_type_acc'] = accuracy_score(y_true=gold_cancer_types, y_pred=pred_cancer_types)
            eval_res['cancer_type_f1'] = f1_score(y_true=gold_cancer_types, y_pred=pred_cancer_types,
                                                  average='weighted')
            eval_res['T_period_acc'] = accuracy_score(y_true=gold_T_periods, y_pred=pred_T_periods)
            eval_res['T_period_f1'] = f1_score(y_true=gold_T_periods, y_pred=pred_T_periods, average='weighted')
            eval_res['N_period_acc'] = accuracy_score(y_true=gold_N_periods, y_pred=pred_N_periods)
            eval_res['N_period_f1'] = f1_score(y_true=gold_N_periods, y_pred=pred_N_periods, average='weighted')

            with open('output/retrieval/retrieval_generation_results.txt', "a", encoding='utf-8') as writer:
                print("***** {} top {} results *****".format(split, top_k))
                writer.write("\n***** {} top {} results *****\n".format(split, top_k))
                for key, value in eval_res.items():
                    print("{} top {}  {} = {}".format(split, top_k, key, value))
                    writer.write("%.2f&" % (value*100))
                    # writer.write("{} top {} {} = {}\n".format(split, top_k, key, value))
