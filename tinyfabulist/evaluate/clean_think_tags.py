import json
import re

input_file = '/home/ap/Documents/Work/Research/tiny_fabulist/data/fables/deepseek-r1-distill-llama-8b-dmb/tf_fables_deepseek-r1-distill-llama-8b-dmb_dt250305-083628.jsonl'
output_file = '/home/ap/Documents/Work/Research/tiny_fabulist/data/fables/clean_deepseek.jsonl'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        fable = data.get('fable', '')
        # Remove everything up to and including </think>
        cleaned_fable = re.sub(r'^.*?</think>', '', fable, flags=re.DOTALL)
        data['fable'] = cleaned_fable.strip()
        outfile.write(json.dumps(data) + '\n')
