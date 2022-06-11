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
    def __init__(self, doc):
        self.doc = doc
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


matcher.add("passive", [pattern_passive])

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
        self.text = sent.text
        self.clauses = []
        self.get_clauses()

    
    def find_conj(self, token, conjs, agent, first_call):
        if first_call:
            pass
        elif token.dep_ != "conj":
            return
        else: pass

        for t in token.children:
            if t.dep_ == "conj":
                conjs.append(t)
                self.find_conj(t, conjs, agent, False)
            elif t.dep_ == "agent":
                agent.append(t)
            else: pass



    def get_clauses(self):
        m = matcher(self.sent)

        for (match_id, token_ids) in m:
            subj, be, pp = [self.sent[i] for i in token_ids[::-1]]
            conjs = []
            conjs.append(pp)
            agent = []
            self.find_conj(pp, conjs, agent, True)
            
            # print("pp: ", pp.text, pp.dep_)
            # print("conjs: ", conjs)
            # print("childrens: ", [(c.text, c.dep_) for c in pp.children])
            # print()

            for c in pp.children:
                if c.dep_ == "agent":
                    agent.append(c)
                #self.find_conj(c, conjs)
            
            agent = "" if not agent else "by " + list(agent[0].children)[0].text #agent.text #
            # print(conjs)

            for conj in conjs:
                self.clauses.append(" ".join((subj.text, be.text, conj.text, agent)))
            

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
        data = read_file(dir+"\\"+filename)
        doc = nlp(data)
        texts.append(doc)
        #texts.append(Text(dir+"\\"+filename, nlp))
    return texts
