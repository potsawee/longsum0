import pickle
import numpy as np
from nltk import tokenize
from rouge import Rouge
rouge_pltrdy = Rouge()

from podcast_processor import PodcastEpisode
from arxiv_processor   import ResearchArticle

class PodcastEpisodeXtra(PodcastEpisode):
    def __init__(self, podcast_id, transcription, description, ext_label_path):
        super().__init__(podcast_id, transcription, description)
        # In addition to podcast_id, transcription, description
        # it adds tran_split, num_sent, extractive label
        self.tran_split = tokenize.sent_tokenize(transcription)
        self.num_sent   = len(self.tran_split)
        self.ext_label  = read_ext_label(ext_label_path)

class ResearchArticleXtra(ResearchArticle):
    def __init__(self, article_id, abstract_text, article_text, ext_label_path):
        super().__init__(article_id, abstract_text, article_text)
        self.num_sent   = len(abstract_text)
        self.ext_label  = read_ext_label(ext_label_path)

def read_ext_label(ext_label_path):
    # ext_label_path = "{}/{}_ext_label.txt".format(i)
    with open(ext_label_path, 'r') as f:
        line = f.read()
    ext_label = [int(x) for x in line.split(',')]
    return ext_label

def f1_oracle_generate(sentences, reference):
    # rouge_pltrdy is case sensitive
    reference = reference.lower()
    sentences = [s.lower() for s in sentences]

    N = len(sentences) # num sentences in the input
    keep_idx  = []
    max_score = 0

    for _ in range(N): # at all this grredy algorithm runs N times
        this_round_scores = [0.0 for _ in range(N)]
        for i in range(N): # go through the list one by one
            if i in keep_idx: # skip if the id already found (already assigned 0.0)
                continue
            being_searched_list = keep_idx + [i]
            being_searched_list = sorted(being_searched_list)

            being_searched_sentences = [sentences[kk] for kk in being_searched_list]

            # being_searched_sentences = [None for _ in range(len(being_searched_list))]
            # for j, sent_i in enumerate(being_searched_list):
            #     being_searched_sentences[j] = sentences[sent_i]

            hypothesis = " ".join(being_searched_sentences)
            try:
                rouge_scores = rouge_pltrdy.get_scores(hypothesis, reference)
                rouge2_f1 = rouge_scores[0]['rouge-2']['f']
                this_round_scores[i] = rouge2_f1
            except ValueError:
                pass
                # print("ValueError")
            except RecursionError:
                print("RecursionError")

        max_score_current_round = np.max(this_round_scores)
        if max_score_current_round > max_score:
            first_rank = np.argsort(this_round_scores)[::-1][0]
            keep_idx.append(first_rank)
            max_score = max_score_current_round
        else: #terminate
            print("num sentences found:" , len(keep_idx))
            return sorted(keep_idx)

    print("take all sentences:" , len(keep_idx))
    return sorted(keep_idx)


def generate_ext_label_podcast_data(data_path, output_dir, start_id, end_id):
    with open(data_path, 'rb') as f:
        podcasts = pickle.load(f, encoding="bytes")
    print("len(podcasts) = {}".format(len(podcasts)))

    ids = [x for x in range(start_id, end_id)]

    for i in ids:
        out_path  = "{}/{}_ext_label.txt".format(output_dir, i)
        sentences = tokenize.sent_tokenize(podcasts[i].transcription)
        reference = podcasts[i].description

        keep_idx = f1_oracle_generate(sentences, reference)

        # if found nothing, selecting the top3 longest sentences
        if len(keep_idx) == 0:
            sent_lengths = [len(tokenize.word_tokenize(ssent)) for ssent in sentences]
            keep_idx = np.argsort(sent_lengths)[::-1][:3].tolist()
            keep_idx = sorted(keep_idx) # previously i forgot this!!

        ext_label_string = ",".join([str(j) for j in keep_idx])
        with open(out_path, "w") as f: f.write(ext_label_string)
        print("processed:", out_path)

def generate_ext_label_arxiv_data(data_path, output_dir, start_id, end_id):
    """
    for arxiv & pubmed
    """
    with open(data_path, 'rb') as f:
        articles = pickle.load(f, encoding="bytes")
    print("len(articles) = {}".format(len(articles)))

    ids = [x for x in range(start_id, end_id)]

    for i in ids:
        out_path  = "{}/{}_ext_label.txt".format(output_dir, i)
        sentences = articles[i].article_text
        reference = " ".join(articles[i].abstract_text)

        keep_idx = f1_oracle_generate(sentences, reference)

        # if found nothing, selecting the top3 longest sentences
        if len(keep_idx) == 0:
            sent_lengths = [len(tokenize.word_tokenize(ssent)) for ssent in sentences]
            keep_idx = np.argsort(sent_lengths)[::-1][:3].tolist()
            keep_idx = sorted(keep_idx) # previously i forgot this!!

        ext_label_string = ",".join([str(j) for j in keep_idx])
        with open(out_path, "w") as f: f.write(ext_label_string)
        print("processed:", out_path)
