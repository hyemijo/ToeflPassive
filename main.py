import preprocess


if __name__ == "__main__":
    #texts = preprocess.get_texts("data_toefl")
    """
    # 1. get basic statistics
    for text in texts:
        print(text.get_sentcnt())
        print(text.get_wordcnt())
        print(text.get_wordcnt_per_sent())
        for i, sent in enumerate(text.sents_raw):
            print(i+1, end=". ")
            print(sent)
        print"""

    # 2. get clause information    
    text = preprocess.Text("test_text.txt", preprocess.nlp)
    text.get_sents_obj()
    for i, sent in enumerate(text.sentences):
        print("===============================================")
        for token in sent.sent:
            print(token.text, "(", token.dep_, end=") ")
        print()    

        print()
        print(str(i) +  ". " + sent.sent.text.strip())
        for clause in sent.clauses:
            print(clause.clause_span.text.strip())
    print()

    """text = preprocess.Text("test_text.txt", preprocess.nlp)
    for sent in text.sents_raw:
        print("sent:" + sent.text)
        for const in sent._.constituents:
            print("const: "+ const.text)
            for t in const:
                print(t.text, t.dep_)
            print()
        print()"""
