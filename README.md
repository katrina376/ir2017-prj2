# IRIE 2017 Project 2

For the course "Information Retrieval and Extraction", Fall semester, 2017\.   Writen by Hao-Yung Chan (Student ID: R04521618)  

## Requirements

Python 3.6 (3.6.3 is used in development)  

## Installation

```
  $ git clone https://github.com/katrina376/ir2017-prj2.git
  $ cd ir2017-prj2
```

## Run

```
  $ python3 feature-base.py <PATH_TO_TRAIN_DATA> <PATH_TO_TEST_DATA> <PATH_TO_SEG_CORPUS> <WIDTH_OF_NOUN_WINDOW> <WIDTH_OF_VERB_WINDOW>
```

Be aware that `<PATH_TO_SEG_CORPUS>` should be the segmented version of the corpus.  

`<WIDTH_OF_NOUN_WINDOW>` and `<WIDTH_OF_VERB_WINDOW>` are the parameter of the model. In the experiment, the odds in the interval of 3–13 are used.  

Make sure that `meta.py` is in the same folder as `feature-base.py`. `meta.py` contains the glossary of POS tags to their simplified version. (Source:[中研院平衡語料庫詞類標記集](http://ckipsvr.iis.sinica.edu.tw/papers/category_list.doc))  

For example, `train.txt`, `test.txt`, `Dream_of_the_Red_Chamber_seg.txt` are all in the same folder as `feature-base.py`, set the width of noun window to 5 and the width of verb window to 7\. Run:  

```
  $ python3 feature-base.py train.txt test.txt Dream_of_the_Red_Chamber_seg.txt 5 7
```
