#encoding: utf-8

from module import part_sentence,distance,part_speech,depend_analysi,cut_clean
from sklearn.feature_extraction.text import TfidfVectorizer


def sentence_process(article):
    sentences = part_sentence(article)#分句
    sentences_all = cut_clean(sentences)#分句清理,["法国 定于 6月 11日 举行 议会 选举 首轮 投票","最新 民调 结果 显示"];
    #words_all = part_sentence_clean_words(sentences)#去除停止词后的格式[["科技日报","北京","6月"],["18日", "新科学家","17日"]]
    vectorized = TfidfVectorizer(max_features=5000)  # 为n-gram计数的稀疏矩阵#？如果单词量超过10000怎么办？？
    X = vectorized.fit_transform(sentences_all)  ###将文本输入，得到词频矩阵,用全文本效果好但每次都执行太慢了
    words_all = []#保存分句、分词后词[["科技日报","北京","6月"],["18日", "新科学家","17日"]]
    for i in range(len(sentences_all)):
        words = sentences_all[i].split()  # 第一句话,[["法国","定于"],["6月","11日"]]
        words_all.append(words)
    # print(words_all)
    article_results =[]
    print(len(sentences_all))
    for a in range(len(words_all)):
        # print(words_all[a])
        #词性标注
        postags_a = part_speech(words_all[a])
        #依存句法分析
        article_result = depend_analysi(a,words_all,words_all[a],postags_a,sentences,X)
        if article_result :
            article_results.append(article_result)
    return sentences_all,article_results#返回每篇文章分句清理


