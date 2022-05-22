import heapq
import json
from collections import defaultdict
from typing import Union

from flask import Flask, request, jsonify
import gensim
import jieba
import numpy as np
from scipy import spatial

app = Flask(__name__)

vocab = gensim.models.KeyedVectors.load_word2vec_format("static/models/sgns.sikuquanshu.word.bz2")


def word_embedding(word: str) -> np.ndarray:
    """
    获得词的向量嵌入表示

    :param word: 词语
    :return: 词语的向量嵌入表示
    """
    return vocab.get_vector(vocab.get_index(word, vocab.get_index("我")))


def words2sentence_embedding(words: Union[list, str]) -> np.ndarray:
    """
    通过句子/词语列表生成句向量

    :param words: 句子/词语列表
    :return: 句向量
    """
    if isinstance(words, str):
        return words2sentence_embedding(list(jieba.cut(words)))
    return np.sum([word_embedding(word) for word in words], axis=0)


def cosine_distance(x: Union[str, np.ndarray], y: Union[str, np.ndarray]) -> np.float64:
    """
    计算 x 与 y 的余弦相似度

    :param x: 句子/向量
    :param y: 向量
    :return: x 与 y 的余弦相似度
    """
    # str & str
    if isinstance(x, str):
        return spatial.distance.cosine(
            words2sentence_embedding(list(jieba.cut(x))),
            words2sentence_embedding(list(jieba.cut(y))))

    # np.ndarray & np.ndarray
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return spatial.distance.cosine(x, y)

    # np.ndarray & str
    return spatial.distance.cosine(x, words2sentence_embedding(list(jieba.cut(y))))


data = defaultdict(list)
with open("static/data/final_data.txt", "r") as f:
    for line in f.readlines():
        line = json.loads(line)
        line["embedding"] = words2sentence_embedding(line["fact"])
        for accusation in line['meta']['accusation']:
            data[accusation].append(line)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


def extract(candidates, text_embedding, num):
    temps = {}
    for i in range(len(candidates)):
        temps[i] = cosine_distance(text_embedding, candidates[i]["embedding"])
    indices = heapq.nlargest(num, temps, temps.get)
    return [candidates[i].copy() for i in indices]


@app.route("/search_samples", methods=['POST'])
def search_samples():
    request_body = json.loads(request.data.decode("utf-8"))  # Request Body
    accusations = request_body["accusations"]
    if len(accusations) > 3:
        return jsonify("The length of accusations is over the limit")
    text = request_body["text"]
    text_embedding = words2sentence_embedding(text)

    results = []

    # 加载用户已选择的罪名条目；目标：搜索出相似度最高的 6 / len(accusation) 条
    if len(accusations) == 1:
        results += extract(data[accusations[0]], text_embedding, 6)
    elif len(accusations) == 2:
        for accusation_name_ in accusations:
            results += extract(data[accusation_name_], text_embedding, 3)
    elif len(accusations) == 3:
        for accusation_name_ in accusations:
            results += extract(data[accusation_name_], text_embedding, 2)

    # 加载用户未选择的罪名条目；目标：搜索出相似度最高的两条
    candidates = []
    for accusation_name_ in data:
        if accusation_name_ in accusations:
            continue
        candidates += data[accusation_name_]
    results += extract(candidates, text_embedding, 2)

    # 去除 embedding 属性（无法 jsonify）
    for item in results:
        del item["embedding"]

    return jsonify(results)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
