import json
import pickle
from tqdm import tqdm

class ResearchArticle(object):
    def __init__(self, article_id, abstract_text, article_text):
        """
          'article_id': str,
          'abstract_text': List[str],
          'article_text': List[str],
        """
        self.article_id = article_id
        self.abstract_text = abstract_text
        self.article_text = article_text


def process(load_path, save_path):
    with open(load_path) as f:
        lines = f.readlines()
    print("len(data):", len(lines))

    articles = [None for _ in range(len(lines))]

    for i, line in tqdm(enumerate(lines)):
        item = json.loads(line)
        article_id = item['article_id']
        abstract_text = item['abstract_text']
        abstract_text = [yy.replace("<S> ", "").replace(" </S>","") for yy in abstract_text]
        article_text = item['article_text']
        article = ResearchArticle(article_id, abstract_text, article_text)
        articles[i] = article

    with open(save_path, "wb") as f:
        pickle.dump(articles, f)

    print("dump:", save_path)

if __name__ == "__main__":

    PATH_TO_DOWNLOAD = '/home/alta/summary/pm574/data' # put where arXiv & pubmed are downloaded here

    if PATH_TO_DOWNLOAD is None:
        raise Exception("PATH_TO_DOWNLOAD is None, specify where arXiv & pubmed are downloaded here")

    # --------- arXiv ---------- #
    arxiv_train_path = "{}/arXiv/arxiv-dataset/train.txt".format(PATH_TO_DOWNLOAD) # 203037
    arxiv_val_path   = "{}/arXiv/arxiv-dataset/val.txt".format(PATH_TO_DOWNLOAD)   # 6436
    arxiv_test_path  = "{}/arXiv/arxiv-dataset/test.txt".format(PATH_TO_DOWNLOAD)  # 6440
    process(arxiv_test_path,  "data/arxiv/arxiv_test.pk.bin")
    process(arxiv_val_path,   "data/arxiv/arxiv_val.pk.bin")
    process(arxiv_train_path, "data/arxiv/arxiv_train.pk.bin")

    # --------- PubMed --------- #
    pubmed_train_path = "{}/PubMed/pubmed-dataset/train.txt".format(PATH_TO_DOWNLOAD)  # 119924
    pubmed_val_path   = "{}/PubMed/pubmed-dataset/val.txt".format(PATH_TO_DOWNLOAD)    # 6633
    pubmed_test_path  = "{}/PubMed/pubmed-dataset/test.txt".format(PATH_TO_DOWNLOAD)   # 6658
    process(pubmed_train_path, "data/pubmed/pudmed_train.pk.bin")
    process(pubmed_val_path,   "data/pubmed/pudmed_val.pk.bin")
    process(pubmed_test_path,  "data/pubmed/pudmed_test.pk.bin")
