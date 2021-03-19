# Identification of informative COVID-19 English Tweets (**WNUT 2020**)
For this task, participants are asked to develop systems that automatically identify whether an English Tweet related to the novel coronavirus (COVID-19) is informative or not. Such informative Tweets provide information about recovered, suspected, confirmed and death cases as well as location or travel history of the cases.


## Data

The dataset consists of 10K COVID English Tweets, including 4719 Tweets labeled as INFORMATIVE and 5281 Tweets labeled as UNINFORMATIVE. Here, each Tweet is annotated by 3 independent annotators and we obtain an inter-annotator agreement score of Fleiss' Kappa at 0.818. We use a 70/10/20 rate for splitting the dataset into training/validation/test sets.

* Training set (released on June 21, 2020): 3303 (INFORMATIVE) & 3697 (UNINFORMATIVE)
* Validation set (released on June 21, 2020): 472 (INFORMATIVE) & 528 (UNINFORMATIVE)
* Test set: 944 (INFORMATIVE) & 1056 (UNINFORMATIVE)

Shared task data is available at: [https://github.com/VinAIResearch/COVID19Tweet](https://github.com/VinAIResearch/COVID19Tweet)

## Evaluation Script

To check efficacy of your model(in development), please use the evaluation script [evaluator.py](https://github.com/VinAIResearch/COVID19Tweet/blob/master/evaluator.py)

## Run

```
python embedding.py
#for using different embeddings (ELMO or XLNET) change in load.py
python model.py

```


## Citation

```
@inproceedings{reddy-biswal-2020-iiitbh,
    title = "{IIITBH} at {WNUT}-2020 Task 2: Exploiting the best of both worlds",
    author = "Reddy, Saichethan  and
      Biswal, Pradeep",
    booktitle = "Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.wnut-1.46",
    doi = "10.18653/v1/2020.wnut-1.46",
    pages = "342--346",
    abstract = "In this paper, we present IIITBH team{'}s effort to solve the second shared task of the 6th Workshop on Noisy User-generated Text (W-NUT)i.e Identification of informative COVID-19 English Tweets. The central theme of the task is to develop a system that automatically identify whether an English Tweet related to the novel coronavirus (COVID-19) is Informative or not. Our approach is based on exploiting semantic information from both max pooling and average pooling, to this end we propose two models.",
}
```
