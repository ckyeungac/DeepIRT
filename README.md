# Deep-IRT
This is the repository for the code in the paper *Deep-IRT: Make Deep Learning Based Knowledge Tracing Explainable Using Item Response Theory* ([EDM](https://drive.google.com/file/d/1iSYGeH0l98HMfdOfGVumigxqZMlQ1t78/view), [arVix](https://arxiv.org/abs/1904.11738))

If you find this repository useful, please cite
```
@inproceedings{EDM2019_Yeung_DeepIRT,
  title={Addressing two problems in deep knowledge tracing via prediction-consistent regularization},
  author={Yeung, Chun Kit},
  year={2019},
  booktitle = {{Proceedings of the 12th International Conference on Educational Data Mining}},
  pages = {683--686}
}
```

## Abstact
Knowledge tracing is one of the key research areas for empowering personalized education. It is a task to model students' mastery level of a knowledge component (KC) based on their historical learning trajectories. In recent years, a recurrent neural network model called deep knowledge tracing (DKT) has been proposed to handle the knowledge tracing task and literature has shown that DKT generally outperforms traditional methods. However, through our extensive experimentation, we have noticed two major problems in the DKT model. The first problem is that the model fails to reconstruct the observed input. As a result, even when a student performs well on a KC, the prediction of that KC's mastery level decreases instead, and vice versa. Second, the predicted performance across time-steps is not consistent. This is undesirable and unreasonable because student's performance is expected to transit gradually over time. To address these problems, we introduce regularization terms that correspond to \emph{reconstruction} and \textit{waviness} to the loss function of the original DKT model to enhance the consistency in prediction. Experiments show that the regularized loss function effectively alleviates the two problems without degrading the original task of DKT.

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