import preprocess
import pickle



if __name__ == "__main__":
    # 0. preprocess toefl texts
    # texts = preprocess.get_texts("data_toefl")
    # f = open("data_pickle.txt", 'wb')
    # pickle.dump(texts, f)
    # f.close()

    # 1. get data
    f = open("data_pickle.txt", 'rb')
    data = pickle.load(f)
    texts = []
    for text in data:
        text = preprocess.Text(text)
        text.get_sents_obj()
        texts.append(text)

    ###############
    # for sample data
    # data = preprocess.read_file("test_text.txt")
    # text = preprocess.Text(preprocess.nlp(data))
    # text.get_sents_obj()
    # texts = []
    # texts.append(text)
    ###############

    ###############
    # get basic statistics
    # for text in texts:
    #     print(text.get_sentcnt())
    #     print(text.get_wordcnt())
    #     print(text.get_wordcnt_per_sent())
    #     for i, sent in enumerate(text.sents_raw):
    #         print(i+1, end=". ")
    #         print(sent)
    #     print
    ###############

    # 2. get clause information    
    # tags: https://web.archive.org/web/20190206204307/https://www.clips.uantwerpen.be/pages/mbsp-tags
    # output = open("passives_full.txt", 'w', encoding="utf-8")
    output = open("clauses_full.txt", 'w', encoding="utf-8")

    cnts = []
    deps = []
    
    for k, text in enumerate(texts, start=1):
        cnt = 0
        for i, s in enumerate(text.sentences, start = 1):
            s_str = s.text.strip().replace("\t", " ")
            tags = [(t.text, t.dep_, t.tag_) for t in s.sent]
            deps.extend(s.vdep)
            print(str(k) + "-" + str(i), s_str, tags, sep="\t", end = "\t", file = output)
            
            if s.clauses:
                cnt += len(s.clauses)
                for j, c in enumerate(s.clauses, start=1):
                    endstr = "" if j == len(s.clauses) else "\t\t\t"
                    print(str(j), c.text, c.dep_, c.tag_, sep="\t", end = "\n"+ endstr, file = output)
            else:
                print(file = output)
        cnts.append(cnt)
    print(cnts)

    # cnts = []
    
    # for k, text in enumerate(texts, start=1):
    #     cnt = 0
    #     for i, s in enumerate(text.sentences, start = 1):
    #         s_str = s.text.strip().replace("\t", " ")
    #         # print(str(k) + "-" + str(i), s_str, sep="\t", end = "\t", file = output)
            
    #         if s.clauses:
    #             cnt += len(s.clauses)
    #             for j, c in enumerate(s.clauses, start=1):
    #                 endstr = "" if j == len(s.clauses) else "\t\t"
    #                 # print(str(j), c, sep="\t", end = "\n"+ endstr, file = output)
    #         # else:
    #             # print(file = output)
    #     cnts.append(cnt)
    # print(cnts)


    ################################################################
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