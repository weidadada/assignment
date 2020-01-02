import re
import math
from scipy.spatial.distance import cosine
# import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from pyltp import SentenceSplitter
from pyltp import Postagger
from pyltp import Parser
from pyltp import Segmentor
from pyltp import NamedEntityRecognizer
import os
# LTP_DATA_DIR = '/home/student/project/project-01/ltp_data'#ltp模型目录的路径
LTP_DATA_DIR = './ltp_data_v3.4.0'#ltp模型目录的路径

say = ['诊断', '交代', '说', '说道', '指出','报道','报道说','称', '警告',
           '所说', '告诉', '声称', '表示', '时说', '地说', '却说', '问道', '写道',
           '答道', '感叹', '谈到', '说出', '认为', '提到', '强调', '宣称', '表明',
           '明确指出', '所言', '所述', '所称', '所指', '常说', '断言', '名言', '告知',
           '询问', '知道', '得知', '质问', '问', '告诫', '坚称', '辩称', '否认', '还称',
           '指责', '透露', '坦言', '表达', '中说', '中称', '他称', '地问', '地称', '地用',
           '地指', '脱口而出', '一脸', '直说', '说好', '反问', '责怪', '放过', '慨叹', '问起',
           '喊道', '写到', '如是说', '何况', '答', '叹道', '岂能', '感慨', '叹', '赞叹', '叹息',
           '自叹', '自言', '谈及', '谈起', '谈论', '特别强调', '提及', '坦白', '相信', '看来',
           '觉得', '并不认为', '确信', '提过', '引用', '详细描述', '详述', '重申', '阐述', '阐释',
           '承认', '说明', '证实', '揭示', '自述', '直言', '深信', '断定', '获知', '知悉', '得悉',
           '透漏', '追问', '明白', '知晓', '发觉', '察觉到', '察觉', '怒斥', '斥责', '痛斥', '指摘',
           '回答', '请问', '坚信', '一再强调', '矢口否认', '反指', '坦承', '指证', '供称', '驳斥',
           '反驳', '指控', '澄清', '谴责', '批评', '抨击', '严厉批评', '诋毁', '责难', '忍不住',
           '大骂', '痛骂', '问及', '阐明']
#初始化
segmentor = Segmentor()#分词
# postagger = Postagger()#词性标注
recognizer = NamedEntityRecognizer()#命名主体识别
# parser = Parser()#依存分析

# pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
# par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
# cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
# ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')
# segmentor.load_with_lexicon(cws_model_path, 'lexicon') # 可自定义单词，加载模型，参数./lexicon是自定义词典的文件路径

# segmentor.load(cws_model_path)
# postagger.load(pos_model_path)
# recognizer.load(ner_model_path)
# parser.load(par_model_path)

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

stop_path = './chinese_stopwords.txt'
def stopwordslist(stop_path):
    stopwords = [line.strip() for line in open(stop_path,encoding='UTF-8-sig').readlines()]
    return stopwords

punc = u'.,;《》？！‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:::「」\\\a-z'
# def clean_words(line):#[["科技日报","北京","6月"],["18日", "新科学家","17日"]]
#     news_content = list(clean(line))
#     # news_content = clean(line)
#     # news_content = news_content.split(" ")
#     STOP_WORDS = stopwordslist(stop_path)
#     for word in news_content:
#         if word in STOP_WORDS:  # 去除停止词
#             news_content.remove(word)
#     return news_content

def clean(line):#["法国 定于 6月 11日 举行 议会 选举 首轮 投票","最新 民调 结果 显示"];
    # we will learn the regular expression next course.
    line_re = re.sub(r"[{}]+".format(punc),"",line)#实现比普通字符串的replace更加强大的替换功能；
    news_content = " ".join(cut(line_re))
    return news_content


# 分句、并清理为每一句["法国 定于 6月 11日 举行 议会 选举 首轮 投票","最新 民调 结果 显示"]
def part_sentence(article):
    sentences = list(SentenceSplitter.split(article.strip()))
    # sentence = SentenceSplitter.split(article)
    # sentence = '\n'.join(sentence)
    # sentences = sentence.split("\n")
    return sentences


def cut_clean(sentences):
    sentences_all = []
    for i in sentences:
        news_content = clean(i)  # 对每一句清理、分词
        sentences_all.append(news_content)
    return sentences_all


# def part_sentence_clean_words(sentences):
#     sentences_all = []
#     for i in sentences:
#         news_content = clean_words(i)#对每一句清理、分词
#         sentences_all.append(news_content)
#     return sentences_all


def tf(word, words):  #document
    # words = document.split()
    word_count = sum(1 for w in words if w == word) + 0.1
    # print(word_count)
    return word_count / len(words)

def document_frequency(word,news_content): #包含该词的文章中数
    return sum(1 for n in news_content if word in n)+1#news_content；+1

#计算IDF
def idf(word,news_content_all):
    """Gets the inversed document frequency"""
    return math.log10(len(news_content_all) / document_frequency(word,news_content_all))


def get_keywords_of_a_ducment(document,news_content_all):#["科技日报","北京","6月"],[["科技日报","北京","6月"],["18日", "新科学家","17日"]]
    words = set(document)#(document.split())#创建一个无序不重复元素集
    #print(words)
    tfidf = [
        (w, tf(w, document) * idf(w,news_content_all)) for w in words
    ]

    tfidf = sorted(tfidf, key=lambda x: x[1], reverse=True)
    return tfidf


def distance(v1, v2): return cosine(v1, v2)#求向量cosine夹角

#词性标注
def part_speech(words):
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    postagger = Postagger() # 初始化实例
    postagger.load(pos_model_path)  # 加载模型
    # postags = postagger.postag(words)  # 词性标注
    # postags = ' '.join(postags)#nh v nt m n u ns n wp
    # postags = postags.split()
    # #print(postags)
    postags = list(postagger.postag(words))
    postagger.release()  # 释放模型
    return postags

#依存句法分析
def depend_analysi(a,words_all,words,postags_a,sentences,X):
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

    parser = Parser() # 初始化实例
    parser.load(par_model_path)  # 加载模型
    arcs = parser.parse(words,postags_a)  # 句法分析

    # rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
    #print(rely_id)
    relation = [arc.relation for arc in arcs]   # 提取依存关系
    #print(relation)
    # heads = ['Root' if id == 0 else words[id-1] for id in rely_id]  # 匹配依存父节点词语

    mixed = [word for word in words if word in say]
    # ne = get_name_entity(sentences[a],postags_a)
    # print(ne)
    name = ''
    stack = []
    c = a +1
    d = a- 1
    # saying = ""
    for k, v in enumerate(arcs ):
        # save the most recent Noun
        if postags_a[k] in ['nh', 'ni', 'ns']:
            stack.append(words[k])
        if v.relation == 'SBV' and (words[v.head - 1] in mixed):  # 确定第一个主谓句
            name = get_name(words[k], words[v.head - 1], words, relation)#, ne)
            saying = get_saying(words, relation, [i.head for i in arcs], v.head)
            print(name)
            if not saying:
                if "“"and "”" in words_all[a-1]:
                    saying = sentences[a-1].strip()

                if "“"and "”" in words_all[a+1]:
                    saying += sentences[a+1].strip()

                if not saying:
                    #与上一句对比
                    p = text_similarity(a-1, X)
                    #与下一句对比
                    z = text_similarity(a, X)
                    if p<z :
                        saying = sentences[a-1].strip()
                        return "在第{}句话中  {}  {}".format(a, name, words[v.head - 1]) + ":{}".format(saying)
                    if p >= z:
                        # saying = sentences[a+1].strip()
                        saying = re.sub(r'[^\w]', '', sentences[a + 1].strip())

                # quotations = re.findall(r'“(.+?)”', sentences[a])#???不明白
                # print(quotations)
                # if quotations:
                #     says = quotations[-1]
                #     print(says)

                for i in range(min(len(sentences) - c - 1,3)):
                    k = text_similarity(c, X)
                    print(k)
                    if (k <= 0.9):
                        # print(saying)
                        saying += sentences[c + 1]
                        # sentences_all[a] = sentences_all[a+1]
                        c += 1
                    else:
                        break
                for i in range(min(d,3)):
                    z = text_similarity_up(d, X)
                    print("up{}".format(z))
                    if (z <= 0.9):
                        # print(saying)
                        saying = sentences[d ] +saying
                        # sentences_all[a] = sentences_all[a+1]
                        d -= 1
                    else:
                        break
            return "在第{}句话中  {}  {}".format(a,name,words[v.head - 1])+":{}".format(saying)
        # 若找到‘：’后面必定为言论。
        if words[k] == '：':
            name = stack.pop()
            saying = ''.join(words[k + 1:])
            return name, saying

    parser.release()
    return False

    # for i in range(len(words)):
    #     #print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')')
    #     if relation[i] == "SBV":#找出"SBV"模型下主语
    #         if heads[i] in say: #如果句子里say[]的词
    #             print("在第{}句话中 {} {}".format(a,words[i],heads[i])+":{}".format(sentences[a+1]))
    #             article_result = "在第{}句话中 {} {}".format(a,words[i],heads[i])+":{}".format(sentences[a+1])
    #             return article_result
    # parser.release()  # 释放模型par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model
    return

#命名实体识别#用不上
def get_name_entity(sentence,postags):
    #sentence = ''.join(strs)
    words = segmentor.segment(sentence)
    # postags = postagger.postag(words) #词性标注
    netags = recognizer.recognize(words, postags) #命名实体识别
    # print(list(netags))
    return netags

# 输入主语第一个词语、谓语、词语数组、词性数组，查找完整主语
def get_name(name, predic, words, property):#, ne):
    index = words.index(name)
    cut_property = property[index + 1:]  # 截取到name后第一个词语
    pre = words[:index]  # 前半部分
    pos = words[index + 1:]  # 后半部分
    # 向前拼接主语的定语
    while pre:
        w = pre.pop(-1)
        w_index = words.index(w)

        if property[w_index] == 'ADV': continue
        if property[w_index] in ['WP', 'ATT', 'SVB'] and (w not in ['，', '。', '、', '）', '（']):
            name = w + name
        else:
            pre = False

    while pos:
        w = pos.pop(0)
        p = cut_property.pop(0)
        if p in ['WP', 'LAD', 'COO', 'RAD'] and w != predic and (w not in ['，', '。', '、', '）', '（']):
            name = name + w  # 向后拼接
        else:  # 中断拼接直接返回
            return name
    return name

# 获取谓语之后的言论
def get_saying(sentence, proper, heads, pos):
    # word = sentence.pop(0) #谓语
    if '：' in sentence:
        return ''.join(sentence[sentence.index('：')+1:])
    while pos < len(sentence):
        w = sentence[pos]
        p = proper[pos]
        h = heads[pos]
        # 谓语尚未结束
        if p in ['DBL', 'CMP', 'RAD']:
            pos += 1
            continue
        # 定语
        if p == 'ATT' and proper[h-1] != 'SBV':
            pos = h
            continue
        # 宾语
        if p == 'VOB':
            pos += 1
            continue
        # if p in ['ATT', 'VOB', 'DBL', 'CMP']:  # 遇到此性质代表谓语未结束，continue
        #    continue

        else:
            if w == '，':
                return ''.join(sentence[pos+1:])
            else:
                return ''.join(sentence[pos:])

def cut(str):
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
    segmentor = Segmentor()   # 初始化实例
    #segmentor.load(cws_model_path)  # 加载模型
    segmentor.load_with_lexicon(cws_model_path, 'lexicon')  # 可自定义单词，加载模型，参数./lexicon是自定义词典的文件路径
    words = segmentor.segment(str)  # 分词,str = "你好，我是大王"
    # print (' '.join(words))
    segmentor.release()  # 释放模型
    return words

#对每一个句子进行文本向量相似比较
def text_similarity(a,X):#['此外   自 本周   6 月 12 日   起   除 小米 手机','印度 的 第一家 小米 之 家 开业   当天']
    # vectorized = TfidfVectorizer(max_features=2000)#为n-gram计数的稀疏矩阵#？如果单词量超过10000怎么办？？
    # X = vectorized.fit_transform(sentences_all)###???将文本sub_samples输入，得到词频矩阵
    print("第{}与第{}句话相似读对比：".format(a,a+1))
    return distance(X[a].toarray()[0], X[a+1].toarray()[0])
    # print(vectorized.vocabulary_)#？？？
    # for i in range(len(sentences_all) - 1):
    #     print(distance(X[0].toarray()[0],X[i+1].toarray()[0]))#值越小越接近

#对每一个句子进行文本向量相似比较
def text_similarity_up(a,X):#['此外   自 本周   6 月 12 日   起   除 小米 手机','印度 的 第一家 小米 之 家 开业   当天']
    # vectorized = TfidfVectorizer(max_features=2000)#为n-gram计数的稀疏矩阵#？如果单词量超过10000怎么办？？
    # X = vectorized.fit_transform(sentences_all)###???将文本sub_samples输入，得到词频矩阵
    print("第{}与第{}句话相似读对比：".format(a, a -1))
    return distance(X[a].toarray()[0], X[a-1].toarray()[0])