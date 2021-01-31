# setup
import requests
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix


class KeySentenceGenerator:

    def __init__(self, p):
        self.path = p

    def gen_ranking(self):
        def by_webpage(p):
            html = requests.get(p).text
            soup = BeautifulSoup(html, features="html.parser")
            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            # get text
            return soup.get_text()

        def by_local(p):
            return open(p, "r").read()

        def clean_txt(tt):
            import re
            sentences = []
            for line in tt.splitlines():
                line_sentences = [st.strip() for st in re.split('[.?!]', line)]
                sentences += [sen for sen in line_sentences if sen != '']
            return sentences

        def clean_sent(single_sent):
            from re import finditer
            pattern = r"[a-z]+('[a-z])?[a-z]*"
            return [match.group(0) for match in finditer(pattern, single_sent.lower())]

        def gen_bag_of_words(sent):
            assert isinstance(sent, list)

            def bag(s0):
                lemmatizer = WordNetLemmatizer()
                temp_bag = []
                for word in clean_sent(s0):
                    for i, j in pos_tag([word]):
                        if j[0].lower() in ['a', 'n', 'v']:
                            temp_bag.append(lemmatizer.lemmatize(word, j[0].lower()))
                        else:
                            temp_bag.append(lemmatizer.lemmatize(word))
                return set(temp_bag) - set(stopwords.words('english'))

            return [bag(s1) for s1 in sent]

        def gen_coords(bag, word_to_id):
            m, n = len(word_to_id), len(bag)
            row, col, val = [], [], []

            from numpy import zeros
            from math import log

            # Compute the `n_i` values
            num_docs = zeros(m)
            for ba in bags:
                for w in ba:
                    num_docs[word_to_id[w]] += 1

            # Construct a coordinate representation
            for j, ba in enumerate(bags):  # loop over each sentence j
                for w in ba:  # loop over each word i in sentence j
                    i = word_to_id[w]
                    a_ij = 1.0 / log((n + 1) / num_docs[i])
                    row.append(i)
                    col.append(j)
                    val.append(a_ij)

            return row, col, val

        def get_svds_largest(a1):
            from scipy.sparse.linalg import svds
            from numpy import abs
            u1, s1, v1 = svds(a1, k=1, which='LM', return_singular_vectors=True)
            return s1, abs(u1.reshape(a.shape[0])), abs(v1.reshape(a.shape[1]))

        def rank_words(u0):
            from numpy import argsort
            return argsort(u0)[::-1]

        def rank_sentences(v0):
            from numpy import argsort
            return argsort(v0)[::-1]

        # Load text data
        sp = self.path.lower()
        if sp.startswith("http://") or sp.startswith("https://"):
            txt = by_webpage(self.path)
        else:
            txt = by_local(self.path)

        # Clean text to make a list of sentences
        sent_lst = clean_txt(txt)

        # Remove stop words and construct bag of words for each sentence
        bags = gen_bag_of_words(sent_lst)

        # Generating IDs, build the sparse matrix a from the bag-of-words representation
        all_words = set()
        for b in bags:
            all_words |= b
        w_t_id = {w: k for k, w in enumerate(all_words)}
        id_t_w = {k: w for k, w in enumerate(all_words)}

        rows, cols, vals = gen_coords(bags, w_t_id)

        a = csr_matrix((vals, (rows, cols)), shape=(len(w_t_id), len(bags)))
        #     # plot sparse martix a
        #     plt.figure(figsize=(9, 9))
        #     plt.spy(A, marker='.', markersize=1)

        # Calculate SVD
        sigma, u, v = get_svds_largest(a)
        #     # plot the entries of u0 and v0
        #     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw=
        #     {'width_ratios': [u0.shape[0], v0.shape[0]]})
        #     ax0.plot(u0, '.')
        #     ax0.set_title('$k$-th entry of $u_0$', loc='left')
        #     ax0.set_xlabel('$k$')
        #     ax1.plot(v0, '.')
        #     ax1.set_title('$k$-th entry of $v_0$', loc='left')
        #     ax1.set_xlabel('$k$')

        # Rank words
        word_ranking = rank_words(u)
        top_ten_words = [id_t_w[k] for k in word_ranking[:10]]
        print("Top 10 words:", top_ten_words)

        # Rank sentences
        sentence_ranking = rank_sentences(v)
        top_five_sentences = [sent_lst[k] for k in sentence_ranking[:5]]

        print("=== Top 5 sentences ===")
        for k, s in enumerate(top_five_sentences):
            # remove non-breaking space if any
            print(f"\n{k}.", repr(s.replace(u'\xa0', u' ')))


if __name__ == '__main__':

    path = "/Users/qiyaowu/Desktop/test.txt"
    KeySentenceGenerator(path).gen_ranking()
