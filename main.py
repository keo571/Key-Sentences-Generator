# setup
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix


def clean_txt(txt):
    import re
    sentences = []
    for line in txt.splitlines():
        line_sentences = [s.strip() for s in re.split('[.?!]', line)]
        sentences += [s for s in line_sentences if s != '']
    return sentences


def clean_sent(single_sent):
    from re import finditer
    pattern = r"[a-z]+('[a-z])?[a-z]*"
    return [match.group(0) for match in finditer(pattern, single_sent.lower())]


def gen_bag_of_words(sent):
    assert isinstance(sent, list)

    def bag(s):
        lemmatizer = WordNetLemmatizer()
        temp_bag = []
        for word in clean_sent(s):
            for i, j in pos_tag([word]):
                if j[0].lower() in ['a', 'n', 'v']:
                    temp_bag.append(lemmatizer.lemmatize(word, j[0].lower()))
                else:
                    temp_bag.append(lemmatizer.lemmatize(word))
        return set(temp_bag) - set(stopwords.words('english'))

    return [bag(s) for s in sent]


def gen_coords(bags, word_to_id):
    m, n = len(word_to_id), len(bags)
    rows, cols, vals = [], [], []

    from numpy import zeros
    from math import log

    # Compute the `n_i` values
    num_docs = zeros(m)
    for b in bags:
        for w in b:
            num_docs[word_to_id[w]] += 1

    # Construct a coordinate representation
    for j, b in enumerate(bags):  # loop over each sentence j
        for w in b:  # loop over each word i in sentence j
            i = word_to_id[w]
            a_ij = 1.0 / log((n + 1) / num_docs[i])
            rows.append(i)
            cols.append(j)
            vals.append(a_ij)

    return rows, cols, vals


def get_svds_largest(A):
    from scipy.sparse.linalg import svds
    from numpy import abs
    u, s, v = svds(A, k=1, which='LM', return_singular_vectors=True)
    return s, abs(u.reshape(A.shape[0])), abs(v.reshape(A.shape[1]))


def rank_words(u0, v0):
    from numpy import argsort
    return argsort(u0)[::-1]


def rank_sentences(u0, v0):
    from numpy import argsort
    return argsort(v0)[::-1]


if __name__ == '__main__':
    # Step 1: get sample data:

    #     # method 0 - through API （take NYT's Most Popular as an example)
    #     api_url = 'https://api.nytimes.com/svc/mostpopular/v2/emailed/7.json?api-key={your key}'
    #     r = requests.get(api_url)
    #     json_data = r.json()
    #     # randomly pick an article
    #     url = json_data['results'][{your pick}]['url']
    #     # # demo
    #     # print(url)
    #     html = requests.get(url).text
    #     soup = BeautifulSoup(html, features="html.parser")
    #     # kill all script and style elements
    #     for script in soup(["script", "style"]):
    #         script.extract()
    #     # get text
    #     text = soup.get_text()

    #     # method 1 - through url (its effectiveness may depend on the html elements on that specific webpage)
    #     url = 'https://www.npr.org/sections/money/2020/12/29/949639548/how-to-make-a-new-years-resolution'
    #     html = requests.get(url).text
    #     soup = BeautifulSoup(html, features="html.parser")
    #     # kill all script and style elements
    #     for script in soup(["script", "style"]):
    #         script.extract()
    #     # get text
    #     text = soup.get_text()

    # method 2 - open local txt file
    text = open("/Users/qiyaowu/PycharmProjects/Key_Sentences_Generator/sample.txt", "r").read()

    # # demo
    # print(text)

    # Step 2: clean text to make a list of sentences
    sent_lst = clean_txt(text)

    # Step 3: remove stop words and construct bag of words for each sentence
    stop_words = stopwords.words('english')
    bags = gen_bag_of_words(sent_lst)

    # Step 4: generating IDs, build the sparse matrix A from the bag-of-words representation
    all_words = set()
    for b in bags:
        all_words |= b
    word_to_id = {w: k for k, w in enumerate(all_words)}
    id_to_word = {k: w for k, w in enumerate(all_words)}

    rows, cols, vals = gen_coords(bags, word_to_id)

    A = csr_matrix((vals, (rows, cols)), shape=(len(word_to_id), len(bags)))
    #     # plot sparse martix A
    #     plt.figure(figsize=(9, 9))
    #     plt.spy(A, marker='.', markersize=1)

    # Step 5: calculate SVD
    sigma0, u0, v0 = get_svds_largest(A)
    #     # plot the entries of u0 and v0
    #     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [u0.shape[0], v0.shape[0]]})
    #     ax0.plot(u0, '.')
    #     ax0.set_title('$k$-th entry of $u_0$', loc='left')
    #     ax0.set_xlabel('$k$')
    #     ax1.plot(v0, '.')
    #     ax1.set_title('$k$-th entry of $v_0$', loc='left')
    #     ax1.set_xlabel('$k$')

    # Step 6: rank words
    word_ranking = rank_words(u0, v0)
    top_ten_words = [id_to_word[k] for k in word_ranking[:10]]
    print("Top 10 words:", top_ten_words)

    # Step 7: rank sentences
    sentence_ranking = rank_sentences(u0, v0)
    top_five_sentences = [sent_lst[k] for k in sentence_ranking[:5]]

    print("=== Top 5 sentences ===")
    for k, s in enumerate(top_five_sentences):
        # remove non-breaking space if any
        print(f"\n{k}.", repr(s.replace(u'\xa0', u' ')))
