# Masters Thesis Project - ReRanking
**SemEval Task 3 - SubTask A (Question-Comment Similarity) : Answer Reranking Problem**

> Given a question and the first 10 comments in its question thread, rerank these 10 comments according to their relevance with respect to the question.

* train.py      : train doc2vec representations
* test.py       : test  doc2vec representations
* myutils.py    : utility library
* pre.py        : preprocess library for 'subtask-A' data
* model.py      : model implementations for 'subtask-A'
* config.json   : json config file (system specific parameters)
* run.sh        : shell script useful for batch runs
* out/          : results folder (output)
* papers/       : base research papers
* test/         : test data and results