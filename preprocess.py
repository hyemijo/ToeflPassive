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


pattern_passive_acl = [
  {
    "RIGHT_ID": "main verb",
    "RIGHT_ATTRS": {"DEP":"acl"} #{"DEP": {"IN" : ["ROOT"]}}
  }
]

pattern_passive_other_cls = [
  {
    "RIGHT_ID": "main verb",
    "RIGHT_ATTRS": {"DEP": {"IN": ["advcl"]}} #{"DEP": {"IN" : ["ROOT"]}} , "relcl"
  }
]


"""
RULES FOR ADVCL PASSIVES
1. check -> advcl only. head verb. (!= noun)
Accepted worldwide by more than 11,500 universities and institutions in over 160 countries, the TOEFL iBT test is the world's premier English-language test for study, work and immigration.
is Accepted by universities


2. overlapping matches
If you don't contact OTI and as a result you can't take the test or your scores are held or cancelled, your test fee will not be refunded.
refunded contact 
refunded take 
refunded held 
refunded cancelled

2. acl x? advcl only?
	The Speaking section is taken at home on your computer monitored online by a human proctor, within 3 days after the paper sections.
taken monitored by proctor

	Check dates cannot be more than 90 days old when received by ETS.
be received by ETS

3. non-passive pp issue. -> past/present perfect tense
"		Wherever you are in your study-abroad journey — whether you've just started to consider your options or you're thinking about the visa application process — we're here to help with the information, online tools and ongoing support you need.
		 are 
're started 
're thinking 
're help 

		Create an account or log in to get started.
log started 

Plan to log in to your account the day before your test to make sure that none of the registration details have changed (for example, a different time or building than originally scheduled).
log make 
time scheduled 

###############
single-word passive exclude?
If you haven't received a response within 90 days, it's likely that you were denied, and you should reach out to the visa office for more information.
's received 

	If you have been granted political asylum, have refugee status or are otherwise unable to meet the ID requirements, you must contact the ETS Office of Testing Integrity (OTI) at least 7 days before you register.
	 granted 
contact register 



RULES FOR ADVCL/RELCL PASSIVES
1. must exclude pp -> have pp, had pp
You can also check with your local test center for a complete list of health and safety procedures they've implemented.
procedures implemented 

2. parsing errors
Your responses are recorded and sent to ETS, where they will be scored by a combination of AI scoring and certified human raters to ensure fairness and quality.
ETS scored by combination
ETS certified by combination

# 3. resolve overlapping matches
Under this provision, a university may issue a Confirmation of Acceptance for Studies (CAS) for students with scores from an English test that is not formally recognized by the U.K. Home Office as a Secure English Language Test (SELT), including the TOEFL iBT test.
test recognized by Office

	Demonstrate your English proficiency with the test that is preferred by 9 out of 10 universities in the United States and accepted everywhere.
test preferred by universities
test accepted by universities


4. 	have, get pp
You must submit your request and have your accommodations approved by ETS Disability Services before your test can be scheduled.
test be scheduled 
accommodations approved by Services



############################################
RULES FOR MAIN CLAUSE PASSIVES
1. passives as a conjunction to the active verb (include -> limited, conj)
Items that may be inspected and/or prohibited include, but are not limited to	neckties

2. fix issues
If you haven't checked in by that time, your test will be canceled and your fee won't be refunded.
test be canceled 
test be refunded 
fee be refunded 
"""

matcher.add("passive", [pattern_passive])
matcher2.add("acl", [pattern_passive_acl])
matcher3.add("other cls", [pattern_passive_other_cls])

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
        # 1. basic passive clause
        # m = matcher(self.sent)
        # for (match_id, token_ids) in m:            
        #     subj, be, pp = [self.sent[i] for i in token_ids[::-1]]
        #     conjs = []
        #     conjs.append(pp)
        #     agent = []
        #     self.find_conj(pp, conjs, agent, True)

        #     for c in pp.children:
        #         if c.dep_ == "agent":
        #             agent.append(c)
            
        #     agent = "" if not agent else "by " + list(agent[0].children)[0].text #agent.text #

        #     for conj in conjs:
        #         self.clauses.append(" ".join((subj.text, be.text, conj.text, agent)))


        # # 2. acl
        # m2 = matcher2(self.sent)

        # for (match_id, token_ids) in m2:
        #     acl = self.sent[token_ids[0]]
        #     passive_acl_flag = acl.tag_ == "VBN" and acl.head.tag_.startswith("NN") # Penn Treebank II tag set. # pp and acl's head noun
        #     if not passive_acl_flag:
        #         continue

        #     head_noun = acl.head
        #     conjs = []
        #     conjs.append(acl)
        #     agent = []
        #     self.find_conj(acl, conjs, agent, True)

        #     for c in acl.children:
        #         if c.dep_ == "agent":
        #             agent.append(c)
            
        #     agent = "" if not agent else "by " + list(agent[0].children)[0].text #agent.text #

        #     for conj in conjs:
        #         self.clauses.append(" ".join((head_noun.text, conj.text, agent)))
            

        # 3. other cls advcl, relcl
        m3 = matcher3(self.sent)

        for (match_id, token_ids) in m3:
            acl = self.sent[token_ids[0]]
            #passive_acl_flag = acl.tag_ == "VBN" and acl.head.tag_.startswith("NN") # Penn Treebank II tag set. # pp and acl's head noun
            #if not passive_acl_flag:
            #    continue

            head_noun = acl.head
            conjs = []
            conjs.append(acl)
            agent = []
            self.find_conj(acl, conjs, agent, True)

            for c in acl.children:
                if c.dep_ == "agent":
                    agent.append(c)
            
            agent = "" if not agent else "by " + list(agent[0].children)[0].text #agent.text #

            for conj in conjs:
                self.clauses.append(" ".join((head_noun.text, conj.text, agent)))    




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
