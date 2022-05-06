import preprocess

if __name__ == "__main__":
    texts = preprocess.get_texts("data_toefl")
    for text in texts:
        print(text.get_sentcnt())
        print(text.get_wordcnt())
        print(text.get_wordcnt_per_sent())
        """for i, sent in enumerate(text.sents_raw):
            print(i+1, end=". ")
            print(sent)"""
        print()