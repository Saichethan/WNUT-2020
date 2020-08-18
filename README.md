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

