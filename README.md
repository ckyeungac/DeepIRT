# Deep-IRT
This is the repository for the code in the paper *Deep-IRT: Make Deep Learning Based Knowledge Tracing Explainable Using Item Response Theory* ([EDM](https://drive.google.com/file/d/1iSYGeH0l98HMfdOfGVumigxqZMlQ1t78/view), [arXiv](https://arxiv.org/abs/1904.11738))

If you find this repository useful, please cite
```
@inproceedings{EDM2019_Yeung_DeepIRT,
  title={Deep-IRT: Make deep learning based knowledge tracing explainable using item response theory},
  author={Yeung, Chun Kit},
  year={2019},
  booktitle = {{Proceedings of the 12th International Conference on Educational Data Mining}},
  pages = {683--686}
}
```

## Abstact
Deep learning based knowledge tracing model has been shown to outperform traditional knowledge tracing model without the need for human-engineered features, yet its parameters and representations have long been criticized for not being explainable. In this paper, we propose Deep-IRT which is a synthesis of the item response theory (IRT) model and a knowledge tracing model that is based on the deep neural network architecture called dynamic key-value memory network (DKVMN) to make deep learning based knowledge tracing explainable. Specifically, we use the DKVMN model to process the student's learning trajectory and estimate the student ability level and the item difficulty level over time. Then, we use the IRT model to estimate the probability that a student will answer an item correctly using the estimated student ability and the item difficulty. Experiments show that the Deep-IRT model retains the performance of the DKVMN model, while it provides a direct psychological interpretation of both students and items.

## Requirements
I have used tensorflow to develop the deep knowledge tracing model, and the following is the packages I used:
```
tensorflow==1.13.2 (or tensorflow-gpu==1.13.2)
scikit-learn==0.21.3
scipy==1.3.1
numpy==1.16.4
```

You can simply install the dependency via
```
pip install -r requirements.txt
```

## Data Format
The first line the number of exercises a student attempted. The second line is the exercise tag sequence. The third line is the response sequence.
```
15
1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
```

## Program Usage
### Run the experiment
```python
python main.py
```

or

```python
python run_experiment.py
```

### Detail hyperparameter for the program
```
usage: main.py [-h] [--dataset DATASET] [--save SAVE] [--cpu CPU]
               [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE] [--train TRAIN]
               [--show SHOW] [--learning_rate LEARNING_RATE]
               [--max_grad_norm MAX_GRAD_NORM]
               [--use_ogive_model USE_OGIVE_MODEL] [--seq_len SEQ_LEN]
               [--n_questions N_QUESTIONS] [--data_dir DATA_DIR]
               [--data_name DATA_NAME] [--memory_size MEMORY_SIZE]
               [--key_memory_state_dim KEY_MEMORY_STATE_DIM]
               [--value_memory_state_dim VALUE_MEMORY_STATE_DIM]
               [--summary_vector_output_dim SUMMARY_VECTOR_OUTPUT_DIM]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     'assist2009', 'assist2015', 'statics2011',
                        'synthetic', 'fsai'
  --save SAVE
  --cpu CPU
  --n_epochs N_EPOCHS
  --batch_size BATCH_SIZE
  --train TRAIN
  --show SHOW
  --learning_rate LEARNING_RATE
  --max_grad_norm MAX_GRAD_NORM
  --use_ogive_model USE_OGIVE_MODEL
  --seq_len SEQ_LEN
  --n_questions N_QUESTIONS
  --data_dir DATA_DIR
  --data_name DATA_NAME
  --memory_size MEMORY_SIZE
  --key_memory_state_dim KEY_MEMORY_STATE_DIM
  --value_memory_state_dim VALUE_MEMORY_STATE_DIM
  --summary_vector_output_dim SUMMARY_VECTOR_OUTPUT_DIM
```