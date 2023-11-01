import os
import re
import codecs
from pythainlp.util import thai_digit_to_arabic_digit
from pythainlp.tokenize import word_tokenize
import pandas as pd
from tqdm import tqdm


def clean_text(text: str):
    res = text.replace('&nbsp;', '')
    res = res.replace('&quot;', '')
    res = thai_digit_to_arabic_digit(res)
    res = res.replace('\n', '')
    res = re.sub(
        r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
        "", res)
    res = re.sub(r'[A-Za-z0-9]*@[A-Za-z]*\.?[A-Za-z0-9]*', "", res)  # romove email
    res = re.sub('!', '', res)
    res = re.sub('\(', '', res)
    res = re.sub('\)', '', res)
    res = re.sub('\[', '', res)
    res = re.sub('\]', '', res)
    res = re.sub('_', '', res)
    res = re.sub('\\\\', '', res)
    res = re.sub('/', '', res)
    res = re.sub('\+', '', res)
    res = re.sub('\=', '', res)
    res = re.sub('\*', '', res)
    res = re.sub(';', '', res)
    res = re.sub('\|', '', res)
    res = re.compile(r'\.{2,}').sub('', res)
    res = re.compile(r'\-{2,}').sub('', res)
    res = re.compile(r'\*{2,}').sub('', res)
    res = re.sub('<', '', res)
    res = re.sub('>', '', res)
    res = re.sub(r'\?', "", res)
    res = re.sub(r'"', "", res)
    res = re.sub(r':', "", res)
    res = re.sub(r'}', "", res)
    res = re.sub(r'{', "", res)
    res = re.sub('—', '', res)
    res = re.sub('·', '', res)
    res = re.sub('•', '', res)
    res = re.sub('“', '', res)
    res = re.sub('”', '', res)
    res = re.sub('‘', '', res)
    res = re.sub('’', '', res)
    res = re.sub('«', '', res)
    res = re.sub('»', '', res)
    res = re.sub(' +', ' ', res)
    res = res.replace('&lt;', '')
    res = res.replace('&#60;', '')
    res = res.replace('&#38;', '')
    res = re.sub(',', '', res)
    res = re.sub('⋯', '', res)
    res = res.replace('\t', '')
    res = re.sub('–', '', res)
    res = re.sub(':', '', res)
    res = res.replace(' . ', '')
    res = re.sub('อ่านต่อที่', '', res)
    res = re.sub('∙', '', res)
    res = res.replace('\u200b', '')
    res = re.sub('…', '', res)
    res = re.sub('#', '', res)
    res = re.sub('-', '', res)
    res = re.compile(r'\+{2,}').sub('', res)

    return remove_emoji(res)


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def tokenize_text(text: str, is_body: bool = True):
    cleaned_text = clean_text(text)
    tokenized_texts = word_tokenize(cleaned_text, engine='attacut')

    if is_body:
        if len(tokenized_texts) <= 512:
            res = ' '.join(tokenized_texts)
            res = ' '.join(res.split(' '))
            res = res.replace(' . ', '. ')
            res = res.replace(' ฯ ', 'ฯ ')
            res = re.sub(' +', ' ', res)
            res = re.sub(r'(?<=\d\d\.)\s(?=\d\d)', '', res)
            res = re.sub(r'(?<=\D.\s)-(?=\D)', '- ', res)
            res = re.sub(r'(?<=\D.\s)-(?=\D)', ' -', res)
            return res.strip()
        else:
            sentences = []
            i = 0
            merge_sentences = []
            for text in tokenized_texts:
                if len(text.strip()) > 0:
                    i = i + 1
                    merge_sentences.append(text)
                    if i == 500:
                        sentences.append(' '.join(merge_sentences))
                        i = 0
                        merge_sentences.clear()
            sentences.append(' '.join(merge_sentences))

            res = '\n'.join(sentences)
            res = ' '.join(res.split(' '))
            res = res.replace(' . ', '. ')
            res = res.replace(' ฯ ', 'ฯ ')
            res = re.sub(' +', ' ', res)
            res = re.sub(r'(?<=\d\d\.)\s(?=\d\d)', '', res)
            res = re.sub(r'(?<=\D.\s)-(?=\D)', '- ', res)
            res = re.sub(r'(?<=\D.\s)-(?=\D)', ' -', res)
            return res.strip()
    else:
        res = ' '.join(tokenized_texts)
        res = ' '.join(res.split(' '))
        res = res.replace(' . ', '. ')
        res = res.replace(' ฯ ', 'ฯ ')
        res = re.sub(' +', ' ', res)
        res = re.sub(r'(?<=\d\d\.)\s(?=\d\d)', '', res)
        res = re.sub(r'(?<=\D.\s)-(?=\D)', '- ', res)
        res = re.sub(r'(?<=\D.\s)-(?=\D)', ' -', res)
        return res.strip()


if __name__ == '__main__':
    df = pd.read_csv('thaipost.csv', encoding='utf-8')
    output_dir = 'thaipost'
    base = 'thaipost_{}'

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if index % 5000 == 0:
            file_name = os.path.join(output_dir, f'{base.format(index)}.txt')
        title = ''
        body = ''

        if type(row['title']) is str:
            title = tokenize_text(row['title'], is_body=False)
            if len(title) > 0:
                title = f'{title}\n'
        if type(row['body']) is str:
            body = tokenize_text(row['body'], is_body=True)
            if len(body) > 0:
                body = f'{body}\n'
        output = f'{title}{body}'

        with codecs.open(file_name, "a", encoding='utf-8') as file_object:
            file_object.write(output)
