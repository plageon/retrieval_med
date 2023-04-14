import json

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

if __name__ == '__main__':
    k_shot = 1

    with open('data/bladder_cancer_train.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cnt_cancer_type = {k: 0 for k, v in cancer_types.items()}
    cnt_T_period = {k: 0 for k, v in T_periods.items()}
    cnt_N_period = {k: 0 for k, v in N_periods.items()}
    data_lines_k_shot = []
    for line in lines:
        line_data = json.loads(line)
        add_line = False
        if cnt_cancer_type[line_data['cancer_type']] < k_shot:
            add_line = True
            cnt_cancer_type[line_data['cancer_type']] += 1
        if cnt_T_period[line_data['T_period']] < k_shot:
            add_line = True
            cnt_T_period[line_data['T_period']] += 1
        if cnt_N_period[line_data['N_period']] < k_shot:
            add_line = True
            cnt_N_period[line_data['N_period']] += 1
        if add_line:
            data_lines_k_shot.append(line)

    with open("data/bladder_cancer_train_{}_shot.jsonl".format(k_shot), "w", encoding="utf-8") as f:
        for line in data_lines_k_shot:
            f.write(line)
