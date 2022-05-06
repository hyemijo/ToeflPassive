#데이터 개수, 표본 평균, 분산, 표준 편차, 다섯 수치 요약 (최솟값, 중간값, 최댓값, 분위수), 최댓값
#-총 단어 수 (어절 수. 문장 수. 수동태, 능동태 수. 절clause 수) -> 추가로?
#-수동태의 비율. -> 선행연구와 부합하는지 확인.

import os
from spacy.language import Language
import spacy


# 0. initialize
@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == "?":
            doc[token.i + 1].is_sent_start = True
    return doc

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")
nlp.add_pipe("set_custom_boundaries", before="parser")

class Text:
    def __init__(self, filename, nlp):
        self.filename = filename
        self.data = read_file(self.filename)
        self.nlp = nlp
        self.doc = self.nlp(self.data)
        self.sents_raw = list(self.doc.sents)

    def get_sentcnt(self):
        return len(self.sents_raw)


    def get_wordcnt(self):
        words  = [word for word in self.data.split()]
        return len(words)

    def get_wordcnt_per_sent(self):
        cnt = 0
        for sent in self.sents_raw:
            cnt += len(sent)

        return cnt /  self.get_sentcnt()   





# 1. read txt file
def read_file(filename):
    with open(filename, 'r', encoding = "utf-8") as f:
        data = [sent.strip() for sent in f.readlines()]
        data_str = "\t".join(data)
        return data_str


def get_texts(dir):
    # return the list of the contents of each txt file in the directory
    texts = []

    for filename in os.listdir(dir):
        texts.append(Text(dir+"\\"+filename, nlp))

    return texts





# 2. parse txt into fileString object






# 3. get basic statistics

