#데이터 개수, 표본 평균, 분산, 표준 편차, 다섯 수치 요약 (최솟값, 중간값, 최댓값, 분위수), 최댓값
#-총 단어 수 (어절 수. 문장 수. 수동태, 능동태 수. 절clause 수) -> 추가로?
#-수동태의 비율. -> 선행연구와 부합하는지 확인.

from collections import defaultdict
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
matcher2 = DependencyMatcher(nlp.vocab)
matcher3 = DependencyMatcher(nlp.vocab)
matcher4 = DependencyMatcher(nlp.vocab)
matcher5 = DependencyMatcher(nlp.vocab)
        

# model architecture
#https://spacy.io/models/en/#en_core_web_md
# label scheme
#https://spacy.io/models/en#en_core_web_md-labels

# https://spacy.io/usage/rule-based-matching#dependencymatcher 
#https://explosion.ai/demos/displacy?text=The%20dog%20was%20chased%20and%20hunted%20and%20killed%20by%20the%20man.&model=en_core_web_sm&cpu=1&cph=1
        # active voice -> nsubj/csubj, (aux), (neg), ROOT/conj(head = root), (dobj)
        # active voice __ to v -> aux advcl // aux xcomp
        # passive voice -> nsubjpass/csubjpass, (aux), (neg), auxpass, ROOT/advcl, (agent)
        # stanford p.11


pattern_clause = [
  {
    "RIGHT_ID": "main verb",
    "RIGHT_ATTRS": {"DEP" : {"IN" : ["ROOT", "conj", "acl", "advcl", "relcl", "auxpass"]}}
  }
]

pattern_active_acl = [
  {
    "RIGHT_ID": "acl",
    "RIGHT_ATTRS": {"DEP":"acl"} #{"DEP": {"IN" : ["ROOT"]}}
  }
]

pattern_active_other_cls = [
  {
    "RIGHT_ID": "other_cls",
    "RIGHT_ATTRS": {"DEP": {"IN": ["advcl", "relcl"]}} #{"DEP": {"IN" : ["ROOT"]}} , "relcl"
  }
]


pattern_active_conj = [
  {
    "RIGHT_ID": "main verb",
    "RIGHT_ATTRS": {"DEP": {"IN" : ["ROOT"]}}
  },
  {
    "LEFT_ID": "main verb",
    "REL_OP": ">",
    "RIGHT_ID": "subject",
    "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "csubj"]}},
  },
  {
    "LEFT_ID": "main verb",
    "REL_OP": ">",
    "RIGHT_ID": "pp_conj",
    "RIGHT_ATTRS": {"DEP": "conj"}
  }
]



pattern_passive = [
  {
    "RIGHT_ID": "main verb",
    "RIGHT_ATTRS": {}
  },
  {
    "LEFT_ID": "main verb",
    "REL_OP": ">",
    "RIGHT_ID": "be verb",
    "RIGHT_ATTRS": {"DEP": "auxpass"}
  },
  {
    "LEFT_ID": "main verb",
    "REL_OP": ">",
    "RIGHT_ID": "subject",
    "RIGHT_ATTRS": {"DEP": {"IN": ["nsubjpass", "csubjpass"]}},
  }
]


pattern_passive_acl = [
  {
    "RIGHT_ID": "acl",
    "RIGHT_ATTRS": {"DEP":"acl"} #{"DEP": {"IN" : ["ROOT"]}}
  }
]

pattern_passive_other_cls = [
  {
    "RIGHT_ID": "other_cls",
    "RIGHT_ATTRS": {"DEP": {"IN": ["advcl", "relcl"]}} #{"DEP": {"IN" : ["ROOT"]}} , "relcl"
  }
]


pattern_passive_conj = [
  {
    "RIGHT_ID": "main verb",
    "RIGHT_ATTRS": {"DEP": {"IN" : ["ROOT"]}}
  },
  {
    "LEFT_ID": "main verb",
    "REL_OP": ">",
    "RIGHT_ID": "subject",
    "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "csubj"]}},
  },
  {
    "LEFT_ID": "main verb",
    "REL_OP": ">",
    "RIGHT_ID": "pp_conj",
    "RIGHT_ATTRS": {"DEP": "conj"}
  },
  {
    "LEFT_ID": "pp_conj",
    "REL_OP": ">",
    "RIGHT_ID": "be verb",
    "RIGHT_ATTRS": {"DEP": "auxpass"}
  }
]


matcher.add("passive", [pattern_passive])
matcher2.add("acl", [pattern_passive_acl])
matcher3.add("other cls", [pattern_passive_other_cls])
matcher4.add("conj", [pattern_passive_conj])
matcher5.add("clause", [pattern_clause])


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
        self.vdep = []
        self.get_clauses_full()
        
        # self.get_clauses()

    
    def find_conj(self, token, conjs, agent, first_call):
        if first_call:
            pass
        elif token.dep_ != "conj":
            # 2. fix subj issues
            # If you haven't checked in by that time, your test will be canceled and your fee won't be refunded.
            # test be canceled 
            # test be refunded 
            # fee be refunded 
            return
        else: pass

        for t in token.children:
            if t.dep_ == "conj" and t.tag_ == "VBN":
                conjs.append(t)
                self.find_conj(t, conjs, agent, False)
            elif t.dep_ == "agent":
                agent.append(t)
            else: pass

    def get_clauses_full(self):
        dep_exclude = "aux auxpass dep amod prep acomp ccomp compound intj".split(" ")
        for t in self.sent:
            if t.dep_ in dep_exclude:
                continue
            if t.tag_.startswith("V"):
                self.clauses.append(t)
                self.vdep.append(t)


        # m = matcher5(self.sent)
        # for (match_id, token_ids) in m:
        #     v = self.sent[token_ids[0]]

        #     if not v.tag_.startswith("V"):
        #         # print(self.sent)
        #         # print([(t.text, t.tag_) for t in self.sent])
        #         # print(v)
        #         continue
        #     self.clauses.append(v)
        return



    def get_clauses(self):        
        # 1. basic passive clause
        m = matcher(self.sent)
        for (match_id, token_ids) in m:            
            subj, be, pp = [self.sent[i] for i in token_ids[::-1]]
            conjs = []
            conjs.append(pp)
            agent = []
            self.find_conj(pp, conjs, agent, True)

            for c in pp.children:
                if c.dep_ == "agent" or (c.lemma == "by" and c.dep_ == "prep"): #c.dep_ == "agent": 
                    agent.append(c)
            
            agent = "" if not agent else "by " + list(agent[0].children)[0].text #agent.text #

            for conj in conjs:
                self.clauses.append(" ".join(("main\t", subj.text, be.text, conj.text, agent)))


        # 2. acl
        m2 = matcher2(self.sent)
        for (match_id, token_ids) in m2:
            acl = self.sent[token_ids[0]]
            if acl.tag_ != "VBN" or not acl.head.tag_.startswith("NN"):
                continue

            head_noun = acl.head
            conjs = []
            conjs.append(acl)
            agent = []
            self.find_conj(acl, conjs, agent, True)

            for c in acl.children:
                if c.dep_ == "agent" or (c.lemma == "by" and c.dep_ == "prep"): #c.dep_ == "agent":
                    agent.append(c)
            
            agent = "" if not agent else "by " + list(agent[0].children)[0].text #agent.text #

            for conj in conjs:
                self.clauses.append(" ".join(("acl\t", head_noun.text, conj.text, agent)))
            

        # 3. other cls advcl, relcl        
        m3 = matcher3(self.sent)
        exclude_subj = ("nsubj", "nsubjpass", "csubj", "csubjpass")
        
        for (match_id, token_ids) in m3:
            advcl = self.sent[token_ids[0]]
            has_subj = any(c.dep_ in exclude_subj for c in advcl.children)
            has_get_pp = any((c.lemma_, c.dep_) == ("get", "auxpass") for c in advcl.children)
            has_perfect_pp = any((c.lemma_, c.dep_) == ("have", "aux") for c in advcl.children) and not any((c.lemma_, c.dep_) == ("be", "auxpass") for c in advcl.children)
            if advcl.tag_ != "VBN" or has_subj or has_get_pp or has_perfect_pp:
                continue

            conjs = []
            conjs.append(advcl)
            agent = []
            self.find_conj(advcl, conjs, agent, True)

            for c in advcl.children:
                if c.dep_ == "agent" or (c.lemma == "by" and c.dep_ == "prep"):
                    agent.append(c)
            
            agent = "" if not agent else "by " + list(agent[0].children)[0].text #agent.text #

            for conj in conjs:
                self.clauses.append(" ".join(("cls\t", conj.text, agent)))    

        # 4. passive with active
        m4 = matcher4(self.sent)
        for (match_id, token_ids) in m4:          
            root, subj, pp, be = [self.sent[i] for i in token_ids]
            if any("subjpass" in t.dep_ for t in pp.children):
                continue

            conjs = []
            conjs.append(pp)
            agent = []
            self.find_conj(pp, conjs, agent, True)

            for c in pp.children:
                if c.dep_ == "agent" or (c.lemma == "by" and c.dep_ == "prep"): #c.dep_ == "agent": 
                    agent.append(c)
            
            agent = "" if not agent else "by " + list(agent[0].children)[0].text #agent.text #

            for conj in conjs:
                self.clauses.append(" ".join(("conj\t", subj.text, be.text, conj.text, agent)))




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
    return texts
