import json

if __name__ == "__main__":
    top_k = 5
    with open("output/retrieval/test_top_{}_res.jsonl".format(top_k), "r", encoding="utf-8") as f:
        top_k_res_lines = f.readlines()

    for line in top_k_res_lines:
        line_data = json.loads(line)
        ref_prompts = "；".join(
            ["“{}”的癌症类型为“{}”".format(ref['text'], ref['cancer_type'], ) for ref in line_data['refs']])
        prompt_cancer_type = "已知：{}。请根据以上信息推断“{}”的癌症类型为：".format(ref_prompts,line_data['text'])
        print(prompt_cancer_type)
        break
