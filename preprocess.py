#데이터 개수, 표본 평균, 분산, 표준 편차, 다섯 수치 요약 (최솟값, 중간값, 최댓값, 분위수), 최댓값
#-총 단어 수 (어절 수. 문장 수. 수동태, 능동태 수. 절clause 수) -> 추가로?
#-수동태의 비율. -> 선행연구와 부합하는지 확인.

from collections import defaultdict
from contextlib import nullcontext
from lib2to3.pgen2 import token
import os
import turtle
from attr import has
from spacy.language import Language
import spacy
import benepar
#benepar.download('benepar_en3')
import nltk
from sympy import false, true
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.strings import StringStore
from spacy import displacy


# 0. initialize
@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == "?":
            doc[token.i + 1].is_sent_start = True
    return doc

nlp = spacy.load('en_core_web_md')
nlp.add_pipe("sentencizer")
nlp.add_pipe("set_custom_boundaries", before="parser")


if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})



# 1. preprocess txt files
class Text:
    def __init__(self, filename, nlp):
        self.filename = filename
        self.data = read_file(self.filename)
        self.nlp = nlp
        self.doc = self.nlp(self.data)
        self.sents_raw = list(self.doc.sents)
        self.sentences = []
    
    # 1.1. get basic stats
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

    # 1.2. get clause info
    def get_sents_obj(self):
        for sent in self.sents_raw:
            self.sentences.append(Sentence(sent))
        return


# 2. preprocess sentences of txt files
from spacy.matcher import DependencyMatcher
matcher = DependencyMatcher(nlp.vocab)


# https://spacy.io/usage/rule-based-matching#dependencymatcher 
#https://explosion.ai/demos/displacy?text=The%20dog%20was%20chased%20and%20hunted%20and%20killed%20by%20the%20man.&model=en_core_web_sm&cpu=1&cph=1
        # active voice -> nsubj/csubj, (aux), (neg), ROOT/conj(head = root), (dobj)
        # active voice __ to v -> aux advcl // aux xcomp
        # passive voice -> nsubjpass/csubjpass, (aux), (neg), auxpass, ROOT/advcl, (agent)
        # stanford p.11

pattern_passive = [
  {
    "RIGHT_ID": "main verb",
    "RIGHT_ATTRS": {} #{"DEP": {"IN" : ["ROOT"]}}
  },

  # be (auxiliary verb)
  {
    "LEFT_ID": "main verb",
    "REL_OP": ">",
    "RIGHT_ID": "be verb",
    "RIGHT_ATTRS": {"DEP": {"IN": ["auxpass"]}}
  },
  
  # subject
  {
    "LEFT_ID": "main verb",
    "REL_OP": ">",
    "RIGHT_ID": "subject",
    "RIGHT_ATTRS": {"DEP": {"IN": ["nsubjpass", "csubjpass"]}},
  }
]


pattern_agent = [
    # by
    {
        "RIGHT_ID": "agent",
        "RIGHT_ATTRS": {"LOWER": "by"}},

    # noun phrase
  {
    "LEFT_ID": "agent",
    "REL_OP": ">",
    "RIGHT_ID": "noun",
    "RIGHT_ATTRS": {"DEP": "pobj"},
  }
]


pattern_conj = [
  {
    "RIGHT_ID": "conj",
    "RIGHT_ATTRS": {"DEP": "conj"}
  },
    # root
  {
    "LEFT_ID": "conj",
    "REL_OP": "<<",
    "RIGHT_ID": "main verb",
    "RIGHT_ATTRS": {"DEP": {"NOT_IN": ["ROOT"]}}
  },
  # be (auxiliary verb)
  {
    "LEFT_ID": "main verb",
    "REL_OP": ">",
    "RIGHT_ID": "be verb",
    "RIGHT_ATTRS": {"DEP": {"IN": ["auxpass"]}}
  }
]


matcher.add("passive", [pattern_passive])
matcher.add("agent", [pattern_agent])
matcher.add("conj", [pattern_conj])

hashs = StringStore(["passive", "agent", "conj"])



# get clause information as a Clause Object
class Clause:
    def __init__(self, clause_span):
        self.clause_span = clause_span
        #self.passive = self.get_voice()
        #self.form = ""
        #self.agent = ""
        #self.patient = ""

    # def get_voice(self):
    #     return False


class Sentence:
    def __init__(self, sent):
        self.sent = sent
        self.clauses = []
        self.get_clauses()

    def get_clauses(self):
        try:
            m = matcher(self.sent)
            print(self.sent)
            for (match_id, token_ids) in m:
                if (match_id == hashs["passive"]):
                    for i, token_id in enumerate(token_ids):
                        t = self.sent[token_id]
                        print(pattern_passive[i]["RIGHT_ID"], t.text)
                        for c in t.children:
                            if c.dep_ == "agent":
                                print(c.text, [k for k in c.children if k.dep_ == "pobj"])



                        #print("\t\t", [t.text for t in t.children])
                    print()

        except:
            print("none")
            print()

        """consts = self.sent._.constituents
        for const in consts:
            if self.is_clause(const):
                clause = Clause(const)
                self.clauses.append(clause)
        return"""

    def get_clause_cnt(self):
        return len(self.clauses)    

    def get_clause_voice(self):
        cnt = {"passive":0, "non-passive":0}
        for clause in self.clauses:
            if clause.passive:
                cnt["passive"] += 1
            else:
                cnt["non-passive"] += 1    
        return cnt
    
    def get_aux_cnt(self):
        cnt = defaultdict(str)
        for clause in self.clauses:
            form = clause.form
            #lemmatize
            cnt[form] += 1
        return cnt


# 3. read txt file
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
