#encoding: utf-8

from flask import Flask,request,session,url_for,current_app,g,render_template,abort,redirect
import flask
from flask_sqlalchemy import SQLAlchemy
import config
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.orm import Session
# from module import get_keywords_of_a_ducment#,clean_words
from test import sentence_process
import pandas as pd  # 用pd.read_sql_query在本地创建数据表

cvs_path = "./project01_db.csv"

app = Flask(__name__)
#添加配置
app.config.from_object(config)
db = SQLAlchemy(app)

class Article(db.Model):  # 一个表就是一个类#创建这个表才能查询，查询已知表内容，比较笨的方法是自己去 show create table tb_user；然后根据表的结构去创建一个User类。
    __tablename__ = 'article'  # 双_受保护的字段
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(100), nullable=False)

#    db.create_all()#创建，映射成立一张表

@app.route('/add/')#结果添加到本地数据库中
def add():
# 增加：
    article1 = Article(title="dddd")
    db.session.add(article1)
    db.session.commit()
    return "你好"

@app.route('/reload/')
def reload():
    engine = create_engine(
        'mysql+mysqlconnector://root:AI@2019@ai@rm-8vbwj6507z6465505ro.mysql.zhangbei.rds.aliyuncs.com:3306/stu_db',
        echo=True)

    # 两种方法：
    # metadata = MetaData(engine)  # 引擎
    # 反射数据库单表
    # use_article = Table('news_chinese', metadata, autoload=True)#, autoload_with=engine)
    # use_article.columns.keys()
    # session = Session(engine)

    # 根据query的语句到conn数据库中查询，并将结果返回给data这个DataFrame
    query = 'select * from news_chinese '
    df = pd.read_sql_query(query, engine)  # 两个参数：检索语句和连接的数据库对象
    df.to_csv(cvs_path)#保存到本地
    return


@app.route('/process/',methods=['GET','POST'])
def process():
    #直接从数据库中读取
    # result_list = session.query(use_article).all()#取出news_chinese表下内容
    # for item in result_list:
    #     if not item.content:continue
    #     print(item.id,item.author,item.source, item.content, item.feature, item.title, item.url)

    #从本地读取
    # content_all = pd.read_csv(cvs_path)
    # content_all.dropna()
    # content_all.reset_index(drop=True)t
    # news_content = clean_words(content_all.content.iloc[i])
    # 清理内容
    # for i in range(5):#len(content_all)):
    #     print(content_all.content.iloc[i])
    #     sentences_all = sentence_process(content_all.content.iloc[i])
    #     text_similarity(sentences_all)


    if flask.request.method == 'GET':
        return flask.render_template('index.html')
    else:
        content = flask.request.form.get('content')
        # print(content)
        sentences_all,article_results  = sentence_process(content)
        # print(article_results)
        # text_similarity(sentences_all)该文章中句子相似度
        content = {
            "article_results" : article_results
        }



    # return flask.redirect(flask.url_for('index'))
    return flask.render_template('process.html',**content)

@app.route('/')
def index():

    return flask.render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
