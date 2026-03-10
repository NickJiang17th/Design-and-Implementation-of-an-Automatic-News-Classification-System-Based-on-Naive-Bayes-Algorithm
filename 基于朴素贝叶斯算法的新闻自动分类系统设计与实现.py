import requests
import re
import time
import jieba
import numpy as np
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

punctuation_pattern = re.compile(r'[\s+\!\/_,$%^*(+\"\')]+|[+——()?【】"？！，。：；、~@#￥%……&*（）]+')

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0'
}



print("爬取'科技'相关新闻（0）...")

results_keyword1 = []
original_titles_keyword1 = []

url = "https://search.sina.com.cn/?q=科技&c=news&page=1"

try:
    r = requests.get(url, headers=headers, timeout=30)
    r.encoding = "utf-8"

    soup = BeautifulSoup(r.text, 'html.parser')
    news = soup.find_all("div", {"class": "box-result"})

    for new in news:
        title_tag = new.find("a")
        if title_tag:
            title = title_tag.text
            title = re.sub(r"\s+", " ", title).strip()
            if title and len(title) > 5:
                results_keyword1.append(title)
                original_titles_keyword1.append(title)

except Exception as e:
    print(f"爬取出错: {e}")

print(f"爬取到 {len(results_keyword1)} 条'科技'相关新闻")




print("爬取'经济'相关新闻（1）...")

results_keyword2 = []
original_titles_keyword2 = []

url = "https://search.sina.com.cn/?q=经济&c=news&page=1"

try:
    r = requests.get(url, headers=headers, timeout=30)
    r.encoding = "utf-8"

    soup = BeautifulSoup(r.text, 'html.parser')
    news = soup.find_all("div", {"class": "box-result"})

    for new in news:
        title_tag = new.find("a")
        if title_tag:
            title = title_tag.text
            title = re.sub(r"\s+", " ", title).strip()
            if title and len(title) > 5:
                results_keyword2.append(title)
                original_titles_keyword2.append(title)

except Exception as e:
    print(f"爬取出错: {e}")

print(f"爬取到 {len(results_keyword2)} 条'经济'相关新闻")




print("爬取'政治'相关新闻（2）...")

results_keyword3 = []
original_titles_keyword3 = []

url = "https://search.sina.com.cn/?q=政治&c=news&page=1"

try:
    r = requests.get(url, headers=headers, timeout=30)
    r.encoding = "utf-8"

    soup = BeautifulSoup(r.text, 'html.parser')
    news = soup.find_all("div", {"class": "box-result"})

    for new in news:
        title_tag = new.find("a")
        if title_tag:
            title = title_tag.text
            title = re.sub(r"\s+", " ", title).strip()
            if title and len(title) > 5:
                results_keyword3.append(title)
                original_titles_keyword3.append(title)

except Exception as e:
    print(f"爬取出错: {e}")

print(f"爬取到 {len(results_keyword3)} 条'政治'相关新闻")

# %% md
# 数据处理并保存到文件
# %%
all_titles_info = []

fo = open("test.txt", "w", encoding="utf-8")
for idx, a in enumerate(results_keyword1):
    a1 = re.sub(r"[\d:,的.]", "", a)
    a2 = jieba.lcut(a1)

    a2_cleaned = []
    for word in a2:
        word_cleaned = punctuation_pattern.sub('', word)
        if word_cleaned:
            a2_cleaned.append(word_cleaned)

    if a2_cleaned:
        fo.write(" ".join(a2_cleaned))
        fo.write(" 0\n")


        original_title = original_titles_keyword1[idx]
        all_titles_info.append({
            'title': original_title,
            'cleaned_data': a2_cleaned,
            'true_label': 0
        })

for idx, a in enumerate(results_keyword2):
    a1 = re.sub(r"[\d:,的.]", "", a)
    a2 = jieba.lcut(a1)

    a2_cleaned = []
    for word in a2:
        word_cleaned = punctuation_pattern.sub('', word)
        if word_cleaned:
            a2_cleaned.append(word_cleaned)

    if a2_cleaned:
        fo.write(" ".join(a2_cleaned))
        fo.write(" 1\n")


        original_title = original_titles_keyword2[idx]
        all_titles_info.append({
            'title': original_title,
            'cleaned_data': a2_cleaned,
            'true_label': 1
        })


for idx, a in enumerate(results_keyword3):
    a1 = re.sub(r"[\d:,的.]", "", a)
    a2 = jieba.lcut(a1)

    a2_cleaned = []
    for word in a2:
        word_cleaned = punctuation_pattern.sub('', word)
        if word_cleaned:
            a2_cleaned.append(word_cleaned)

    if a2_cleaned:
        fo.write(" ".join(a2_cleaned))
        fo.write(" 2\n")


        original_title = original_titles_keyword3[idx]
        all_titles_info.append({
            'title': original_title,
            'cleaned_data': a2_cleaned,
            'true_label': 2
        })

fo.close()


# %% md
# 朴素贝叶斯分类器
# %%
class Naive_bayes_multiclass(object):
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.class_vectors = []
        self.class_probabilities = []
        self.vocab_set = []
        super(Naive_bayes_multiclass, self).__init__()

    def create_vocab_list(self, dataset):
        vocab_set = set([])
        for document in dataset:
            vocab_set = vocab_set | set(document)
        self.vocab_set = list(vocab_set)

    def wordset2vector(self, inputset):
        return_vec = [0] * len(self.vocab_set)
        for word in inputset:
            if word in self.vocab_set:
                return_vec[self.vocab_set.index(word)] += 1
        return return_vec

    def compute_condition_probability(self, words_vec, labels):
        num_train_docs = len(words_vec)
        num_words = len(words_vec[0])

        # 初始化类别计数
        class_counts = []
        for _ in range(self.num_classes):
            class_counts.append(np.ones(num_words))

        # 统计每个类别的单词词频
        for i in range(num_train_docs):
            label = labels[i]
            class_counts[label] += words_vec[i]

        self.class_vectors = []
        for class_id in range(self.num_classes):
            class_vector = np.log(class_counts[class_id] / sum(class_counts[class_id]))
            self.class_vectors.append(class_vector)

        total_samples = len(labels)
        self.class_probabilities = []
        for class_id in range(self.num_classes):
            class_count = sum(1 for label in labels if label == class_id)
            self.class_probabilities.append(np.log(class_count / total_samples))

    def fit(self, dataset, labels):
        self.create_vocab_list(dataset)
        words_vec = []
        for inputset in dataset:
            words_vec.append(self.wordset2vector(inputset))
        self.compute_condition_probability(words_vec, labels)

    def predict(self, word_vec):
        class_scores = []
        for class_id in range(self.num_classes):
            score = sum(word_vec * self.class_vectors[class_id]) + self.class_probabilities[class_id]
            class_scores.append(score)

        return np.argmax(class_scores)


def load_dataset(filename, delimiter=" "):
    dataset = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as fp:
        while True:
            lines = fp.readline().strip()
            if not lines:
                break
            feature = lines.split(delimiter)
            if len(feature) < 2:
                continue
            key = int(feature[-1])
            values = [v.lower() for v in feature[0:-1]]
            labels.append(key)
            dataset.append(values)
    return dataset, labels



def crawl_international_news():

    international_news_urls = [
        "https://news.sina.com.cn/world/",
    ]

    international_titles = []
    all_international_info = []

    for url in international_news_urls:
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.encoding = "utf-8"

            soup = BeautifulSoup(r.text, 'html.parser')

            title_selectors = [
                soup.find_all("h2"),
                soup.find_all("a", {"class": "news-item"}),
                soup.select("ul.news-list li a"),
                soup.select("div.news-item h3"),
                soup.select("div.news-content h2"),
                soup.find_all("div", {"class": "news-item"}),
            ]

            found_titles = False

            for selector_list in title_selectors:
                if selector_list and len(selector_list) > 0:
                    for item in selector_list:
                        if hasattr(item, 'text'):
                            title = item.text.strip()
                            title = re.sub(r"\s+", " ", title).strip()
                            if title and len(title) > 5 and title not in international_titles:
                                international_titles.append(title)

                                a1 = re.sub(r"[\d:,的.]", "", title)
                                a2 = jieba.lcut(a1)

                                a2_cleaned = []
                                for word in a2:
                                    word_cleaned = punctuation_pattern.sub('', word)
                                    if word_cleaned:
                                        a2_cleaned.append(word_cleaned)

                                if a2_cleaned:
                                    all_international_info.append({
                                        'title': title,
                                        'cleaned_data': a2_cleaned,
                                        'source_url': url
                                    })
                                found_titles = True

                    if found_titles:
                        break

            if not found_titles:
                all_links = soup.find_all("a")
                for link in all_links:
                    if hasattr(link, 'text'):
                        title = link.text.strip()
                        if len(title) > 10 and "新闻" not in title and title not in international_titles:
                            international_titles.append(title)
            time.sleep(1)

        except Exception as e:
            continue

    return all_international_info


# %% md
# 主程序
# %%
if __name__ == "__main__":

    filename = "test.txt"
    dataset, labels = load_dataset(filename)

    if len(dataset) == 0:
        print("错误: 没有加载到训练数据")
    else:
        class_names = ['科技', '经济', '政治']
        class_counts = [labels.count(0), labels.count(1), labels.count(2)]

        print(f"\n训练数据统计:")
        for i in range(3):
            print(f"  类别{i}({class_names[i]}): {class_counts[i]} 条")

        naive_bayes = Naive_bayes_multiclass(num_classes=3)
        naive_bayes.fit(dataset, labels)

        international_news = crawl_international_news()

        if international_news:
            print("国际新闻分类:")

            class_counts_test = [0, 0, 0]

            for i, news_info in enumerate(international_news, 1):
                test_vec = naive_bayes.wordset2vector(news_info['cleaned_data'])
                predicted_label = naive_bayes.predict(test_vec)

                print(f"{i}: {news_info['title']}")
                print(f"预测分类为: {predicted_label}")
                print()

                class_counts_test[predicted_label] += 1

            print("分类统计:")
            for i in range(3):
                print(f"类别{i}: {class_counts_test[i]}条")

        else:

            print("\n未能爬取新闻数据")
