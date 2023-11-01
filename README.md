# Saphan-Khwai RoBERTa
Saphankwai Robera base is a Thai language model trained on a 20GB corpus of Thai text collected from diverse online sources, including news and magazines.
#### Disclaimer 
This language model was intentionally trained for the purpose of fine-tuning with a Thai text summarization model. It's important to note that this repository may contain personal data that have been found in the training data. Please use it with caution and at your own risk.

#### Interested in Dataset?
For educational purposes only, if you're interested in the training dataset used in this project, please feel free to reach out to me via email at nakhun.com(at sign)gmail.com.

### Download the Model
[saphankwai_roberta_base.zip](https://nakhun-chumpolsathien.oss-us-west-1.aliyuncs.com/thaisum/saphankwai_roberta_base.zip)

### Notes
- The training set was tokenized at the word-level using ThaiNLP.
- The model underwent 1,000,000 training steps, with the published checkpoint available at the 950,000th step.
- This model was trained on a single RTX 3090 GPU.
- The article title and summary were considered as a single sentence.
- Sentence tokenization of article body was based on the text's length; if the tokenized text exceeded 500 tokens, it was then split into a new sentence.
### Simple Usage
#### Masked Language Modeling

```
import re
from transformers import pipeline
from pythainlp.tokenize import word_tokenize

spk = 'PATH/TO/saphankwai_roberta_base'
fill = pipeline('fill-mask', model=spk , tokenizer=spk )

res = fill(f'ผู้หญิง ประกอบ อาชีพ {fill.tokenizer.mask_token}')
print(res)

 """
[{'score': 0.28720295429229736,
  'token': 733,
  'token_str': ' อะไร',
  'sequence': 'ผู้หญิง ประกอบ อาชีพ อะไร'},
 {'score': 0.18980547785758972,
  'token': 4407,
  'token_str': ' เกษตรกรรม',
  'sequence': 'ผู้หญิง ประกอบ อาชีพ เกษตรกรรม'},
 {'score': 0.06136530637741089,
  'token': 2892,
  'token_str': ' ทนายความ',
  'sequence': 'ผู้หญิง ประกอบ อาชีพ ทนายความ'},
 {'score': 0.032027967274188995,
  'token': 2416,
  'token_str': ' ประมง',
  'sequence': 'ผู้หญิง ประกอบ อาชีพ ประมง'},
 {'score': 0.029685532674193382,
  'token': 2625,
  'token_str': ' การงาน',
  'sequence': 'ผู้หญิง ประกอบ อาชีพ การงาน'}]
 """
 
 
res = fill(f'ผู้ชาย ประกอบ อาชีพ {fill.tokenizer.mask_token}')
print(res)
"""
[{'score': 0.20459885895252228,
  'token': 733,
  'token_str': ' อะไร',
  'sequence': 'ผู้ชาย ประกอบ อาชีพ อะไร'},
 {'score': 0.08249568939208984,
  'token': 2892,
  'token_str': ' ทนายความ',
  'sequence': 'ผู้ชาย ประกอบ อาชีพ ทนายความ'},
 {'score': 0.07744300365447998,
  'token': 4407,
  'token_str': ' เกษตรกรรม',
  'sequence': 'ผู้ชาย ประกอบ อาชีพ เกษตรกรรม'},
 {'score': 0.05463568493723869,
  'token': 2625,
  'token_str': ' การงาน',
  'sequence': 'ผู้ชาย ประกอบ อาชีพ การงาน'},
 {'score': 0.03933854028582573,
  'token': 752,
  'token_str': ' ตำรวจ',
  'sequence': 'ผู้ชาย ประกอบ อาชีพ ตำรวจ'}]
""" 
```