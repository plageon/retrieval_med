import json

if __name__ == "__main__":
    with open("data/bladder_cancer_dev.jsonl", "r", encoding="utf-8") as f:
        json_lines = f.readlines()

    with open("data/bladder_cancer_dev_queries.tsv", "w", encoding="utf-8") as f:
        for line in json_lines:
            line_data = json.loads(line)
            _id = line_data['id']
            _query = line_data['text']
            f.write('{}\t{}\n'.format(_id, _query))
