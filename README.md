# Key-Sentences-Generator

## Objectives
Implements a procedure using singular value decomposition (SVD) to summarize a long document by identifying and ranking important sentences automatically.

## Requirements
- requests
- bs4
- nltk
- scipy
- matplotlib
- re
- numpy

##  Sample Data
### Input: 
An article named *How to design your ideal diet*, written by Cindy Santa Ana on Fairfax County Times.

### Ouput: 
- Top 10 words: ['food', 'eat', 'diet', 'need', 'feel', 'come', 'way', 'day', 'want', 'go']

- Top 5 sentences:
  1. 'You’ll find hundreds of studies saying that veganism is best or Paleo is the way to go, or everyone should be eating a raw food diet'
  2. 'We’re eating dinner at the drive-thru, snacks come in packages and boxes instead of whole foods and we turn a blind eye to the atrocities of conventional animal feeding operations or CAFO’s'
  3. 'This allows us to support local businesses and farms, reduce our environmental impact and eat food that is freshest – which is going to be great for our health'
  4. 'I’m not saying you should never eat them – however, it’s important to also consider what foods are abundant in your area, and also what’s in season'
  5. 'Jot down how you feel and start to make the connection between food, your mood and energy levels after you eat'

##  Algorithm
The algorithm comes from the book *Matrix Methods in Data Mining and Pattern Recognition* by Lars Eldén.

It is based on the intuitive idea that a sentence is important if it contains many important words. And a word is important if it appears in many important sentences.

### Data Representation
Suppose the document consists of *n* distinct sentences and *m* unique words. Next, let *i* be a word and let *j* be a sentence. Furthermore, let *n<sub>i</sub>* denote the number of sentences containing word *i*. We can represent the document by a matrix A, called the term-sentence matrix, where each entry *a<sub>i,j</sub>* is defined as: 

<img src="https://render.githubusercontent.com/render/math?math=a_%7Bi%2Cj%7D%20%3D%20%0A%5Cbegin%7Bcases%7D%20%0A%5Cfrac%7B1%7D%7B%5Cln(%5Cfrac%7Bn%2B1%7D%7Bn_i%7D)%7D%20%26%20%5Ctext%7Bif%20word%20i%20appears%20in%20sentence%20j%2C%20or%7D%20%5C%5C%0A0%20%26%20%5Ctext%7Botherwise.%7D%0A%5Cend%7Bcases%7D">

*a<sub>i,j</sub>* tends to be small if *i* does not appear in many sentences (i.e., *n<sub>i</sub>* is small). The *n*+1 ensures that we do not divide by zero when calculating *a<sub>i,j</sub>*; in particular, even if word *i* appears in every sentence (*n<sub>i</sub>* = *n*), the value ln ((*n* + 1) / (*n<sub>i</sub>*)) > 0. Each sentence contains just a few of the possible words. Therefore, A is sparse.

Here plot the sparse matrix A:
![alt text](https://github.com/keo571/Key_Sentences_Generator/blob/master/sparse_matrix_A.png?raw=true)

### Mathematical formulation of the task
Word’s importance is *w<sub>i</sub>*, sentence importance is *s<sub>j</sub>*. These scores are inter-related. Suppose these relationships are linear, then *w<sub>i</sub>* is proportional to the sum of *a<sub>i,j</sub>* * *s<sub>j</sub>* for every sentence, *s<sub>j</sub>* is proportional to the sum of *a<sub>i,j</sub>* * *w<sub>i</sub>* for every word.

Figure out the *w<sub>i</sub>* and *s<sub>j</sub>* for every word and every sentence, then the most important words and sentences should have the largest scores.

The above model can be rewritten in matrix form.
Letting *w* = [*w<sub>0</sub>*,*w<sub>1</sub>*,…,*w<sub>m−1</sub>*] be the (column) vector of word scores and *s* = [*s<sub>0</sub>*,*s<sub>1</sub>*,…,*s<sub>n−1</sub>*] be the (column) vector of sentence scores, we can define *w* and *s* as:

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Barray%7D%7Bc%7D%0Aw_i%20%5Cpropto%20%5Csum_j%20a_%7Bi%2Cj%7D%20s_j%20%5C%5C%0As_j%20%5Cpropto%20%5Csum_i%20a_%7Bj%2Ci%7D%20w_i%0A%5Cend%7Barray%7D">

*c<sub>w</sub>* and *c<sub>s</sub>* are two unknown contants.

Going one step further, plug these two eauations into one antoher to obtain the following:

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Barray%7D%7Bc%7D%0A%20%20(A%20A%5ET)%20w%20%3D%20(c_s%20c_w)%20w%20%5C%5C%0A%20%20(A%5ET%20A)%20s%20%3D%20(c_w%20c_s)%20s%0A%5Cend%7Barray%7D">

Now it becomes eigenvalue problems. Using SVD, we can have *s* and *w*. 

SVD takes a rectangular matrix of gene expression data (defined as A, where A is a *n x p* matrix) in which the *n* rows represents the genes, and the *p* columns represents the experimental conditions. The SVD theorem states:

<img src="https://render.githubusercontent.com/render/math?math=A_%7Bn%20%5Ctimes%20p%7D%20%3D%20U_%7Bn%20%5Ctimes%20n%7DS_%7Bn%20%5Ctimes%20p%7DV%5ET_%7Bp%20%5Ctimes%20p%7D">

U and V are orthogonal.

The eigenvectors of A<sup>T</sup>A make up the columns of V , the eigenvectors of AA<sup>T</sup> make up the columns of U. Also, the singular values in S are square roots of eigenvalues from A<sup>T</sup>A or AA<sup>T</sup>.  

We find the largest singular value σ<sub>0</sub> of A, and it left and right singular vectors, then the left singular vector u<sub>0</sub> is *w*, the right vector  v<sub>0</sub> is *s*. 
 
Here plot the u<sub>0</sub> and v<sub>0</sub>:

![alt text](https://github.com/keo571/Key_Sentences_Generator/blob/master/entries_u0&v0.png?raw=true)

Finally, we can use u<sub>0</sub> and v<sub>0</sub> to rank the words and sentences.

## License
[MIT](https://choosealicense.com/licenses/mit/)
