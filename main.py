import preprocess
import pickle



if __name__ == "__main__":
    # 0. preprocess toefl texts
    # texts = preprocess.get_texts("data_toefl")
    # f = open("data_pickle.txt", 'wb')
    # pickle.dump(texts, f)
    # f.close()

    # 1. get data
    # f = open("data_pickle.txt", 'rb')
    # data = pickle.load(f)

    """
    # get basic statistics
    for text in texts:
        print(text.get_sentcnt())
        print(text.get_wordcnt())
        print(text.get_wordcnt_per_sent())
        for i, sent in enumerate(text.sents_raw):
            print(i+1, end=". ")
            print(sent)
        print"""

    # 2. get clause information    
    data = preprocess.read_file("test_text.txt")
    text = preprocess.Text(preprocess.nlp(data))
    text.get_sents_obj()
    for s in text.sentences:
        if s.clauses:
            print(s.text)
            for c in s.clauses:
                print("\t", c)
            print()

    # texts = []
    # for text in data:
    #     texts.append(preprocess.Text(text))

    # sample = texts[0]
    # sample.get_sents_obj()
    # for s in sample.sentences:
    #     if s.clauses:
    #         print(s.text)
    #         for c in s.clauses:
    #             print("\t", c)
    #         print()

    
    


    # for i, sent in enumerate(text.sentences):
    #     print("===============================================")
    #     for token in sent.sent:
    #         print(token.text, "(", token.dep_, end=") ")
    #     print()    

    #     print()
    #     print(str(i) +  ". " + sent.sent.text.strip())
    #     for clause in sent.clauses:
    #         print(clause.clause_span.text.strip())
    # print()