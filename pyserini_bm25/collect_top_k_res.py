import json

if __name__ == "__main__":
    top_k = 10
    split = 'dev'
    with open("output/retrieval/{}_search_res.txt".format(split), "r", encoding="utf-8") as f:
        res_lines = f.readlines()

    data_dict = {}
    with open("data/period_type.json", "r", encoding="utf-8") as f:
        data_lines = json.loads(f.read())
    for item in data_lines:
        data_dict[item['id']] = item
    del data_lines

    top_k_res = []
    last_query_id = ''
    for line in res_lines:
        query_id, _, doc_id, top_n, _, _ = line.split()
        if query_id == last_query_id:
            if int(top_n) <= top_k:
                ref = data_dict[doc_id]
                top_k_res[-1]['refs'].append(ref)
        else:
            last_query_id = query_id
            gold_data = data_dict[query_id]
            ref = data_dict[doc_id]
            gold_data['refs'] = [ref]
            top_k_res.append(gold_data)

    with open("output/retrieval/{}_top_{}_res.jsonl".format(split, top_k), "w", encoding="utf-8") as f:
        for line_data in top_k_res:
            f.write(json.dumps(line_data, ensure_ascii=False) + '\n')
