# Masters Thesis Project - ReRanking
**SemEval Task 3 - SubTask A (Question-Comment Similarity) : Answer Reranking Problem**

> Given a question and the first ten comments in its question thread, rerank 
<br/>the 10 comments according to their relevance with respect to the question.

----

| File              | Description                                             |
|:----------------- |:------------------------------------------------------- |
| train.py          | train doc2vec representations                           |
| test.py           | test  doc2vec representations                           |
| myutils.py        | utility library                                         |
| pre.py            | preprocess library for 'subtask-A' data                 |
| model.py          | model implementations for 'subtask-A'                   |
| features.py       | semantic and metadata features implem.                  |
| postag.py         | helper py script for POS tagging and caching results    |
|                   |                                                         |
| config.json       | json config file (system specific parameters)           |
| tagger_cache.json | json containing pos tags for train-dev-test data        |
|                   |                                                         |
| doc2vec.sh        | shell script for batch runs of doc2vec model training   |
| model.sh          | shell script for batch runs of neuralnet model training |
|                   |                                                         |
| out/              | results folder (output)                                 |
| papers/           | base research papers                                    |
| test/             | test data and results                                   |

# TODO
* Cite SemanticZ for Semantic and Metadata features
* Stanford POS Tagger citation
* Cite SemEval for Datasets (test, train sources might be different)
* Update the thesis Appendix A and (PV-DBOW vs PV-DM)'s PV-DBOW results ?