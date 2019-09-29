import argparse
import datetime
import logging
import numpy as np
import tensorflow as tf
import os
from load_data import DataLoader
from model import DeepIRTModel
from run import run_model
from utils import getLogger
from configs import ModelConfigFactory

# set logger
logger = getLogger('Deep-IRT-model')

# argument parser
parser = argparse.ArgumentParser()
# dataset can be assist2009, assist2015, statics2011, synthetic, fsai
parser.add_argument('--dataset', default='assist2009', 
                    help="'assist2009', 'assist2015', 'statics2011', 'synthetic', 'fsai'")
                    
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--cpu', type=bool, default=False)
parser.add_argument('--n_epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--train', type=bool, default=None)
parser.add_argument('--show', type=bool, default=None)    
parser.add_argument('--learning_rate', type=float, default=None)
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--use_ogive_model', type=bool, default=False)

# parameter for the dataset
parser.add_argument('--seq_len', type=int, default=None)
parser.add_argument('--n_questions', type=int, default=None)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--data_name', type=str, default=None)

# parameter for the DKVMN model
parser.add_argument('--memory_size', type=int, default=None)
parser.add_argument('--key_memory_state_dim', type=int, default=None)
parser.add_argument('--value_memory_state_dim', type=int, default=None)
parser.add_argument('--summary_vector_output_dim', type=int, default=None)

_args = parser.parse_args()
args = ModelConfigFactory.create_model_config(_args)
logger.info("Model Config: {}".format(args))

# create directory
for directory in [args.checkpoint_dir, args.result_log_dir, args.tensorboard_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def train(model, train_q_data, train_qa_data, 
            valid_q_data, valid_qa_data, result_log_path, args):
    saver = tf.train.Saver()
    best_loss = 1e6
    best_acc = 0.0
    best_auc = 0.0
    best_epoch = 0.0

    with open(result_log_path, 'w') as f:
        result_msg = "{},{},{},{},{},{},{}\n".format(
            'epoch', 
            'train_auc', 'train_accuracy', 'train_loss',
            'valid_auc', 'valid_accuracy', 'valid_loss'
        )
        f.write(result_msg)
    for epoch in range(args.n_epochs):
        
        train_loss, train_accuracy, train_auc = run_model(
            model, args, train_q_data, train_qa_data, mode='train'
        )
        valid_loss, valid_accuracy, valid_auc = run_model(
            model, args, valid_q_data, valid_qa_data, mode='valid'
        )

        # add to log
        msg = "\n[Epoch {}/{}] Training result:      AUC: {:.2f}%\t Acc: {:.2f}%\t Loss: {:.4f}".format(
            epoch+1, args.n_epochs, train_auc*100, train_accuracy*100, train_loss
        )
        msg += "\n[Epoch {}/{}] Validation result:    AUC: {:.2f}%\t Acc: {:.2f}%\t Loss: {:.4f}".format(
            epoch+1, args.n_epochs, valid_auc*100, valid_accuracy*100, valid_loss
        )
        logger.info(msg)

        # write epoch result
        with open(result_log_path, 'a') as f:
            result_msg = "{},{},{},{},{},{},{}\n".format(
                epoch, 
                train_auc, train_accuracy, train_loss,
                valid_auc, valid_accuracy, valid_loss
            )
            f.write(result_msg)

        # add to tensorboard
        tf_summary = tf.Summary(
            value=[
                tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                tf.Summary.Value(tag="train_auc", simple_value=train_auc),
                tf.Summary.Value(tag="train_accuracy", simple_value=train_accuracy),
                tf.Summary.Value(tag="valid_loss", simple_value=valid_loss),
                tf.Summary.Value(tag="valid_auc", simple_value=valid_auc),
                tf.Summary.Value(tag="valid_accuracy", simple_value=valid_accuracy),
            ]
        )
        model.tensorboard_writer.add_summary(tf_summary, epoch)
        
        # save the model if the loss is lower
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_acc = valid_accuracy
            best_auc = valid_auc
            best_epoch = epoch+1

            if args.save:
                model_dir = "ep{:03d}-auc{:.0f}-acc{:.0f}".format(
                    epoch+1, valid_auc*100, valid_accuracy*100,
                )
                model_name = "Deep-IRT"
                save_path = os.path.join(args.checkpoint_dir, model_dir, model_name)
                saver.save(sess=model.sess, save_path=save_path)

                logger.info("Model improved. Save model to {}".format(save_path))
            else:
                logger.info("Model improved.")

    # print out the final result
    msg = "Best result at epoch {}: AUC: {:.2f}\t Accuracy: {:.2f}\t Loss: {:.4f}".format(
        best_epoch, best_auc*100, best_acc*100, best_loss
    )
    logger.info(msg)
    return best_auc, best_acc, best_loss

def cross_validation():
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    aucs, accs, losses = list(), list(), list()
    for i in range(5):
        tf.reset_default_graph()
        logger.info("Cross Validation {}".format(i+1))
        result_csv_path = os.path.join(args.result_log_dir, 'fold-{}-result'.format(i)+'.csv')

        with tf.Session(config=config) as sess:
            data_loader = DataLoader(args.n_questions, args.seq_len, ',')
            model = DeepIRTModel(args, sess, name="Deep-IRT")
            sess.run(tf.global_variables_initializer())
            if args.train:
                train_data_path = os.path.join(args.data_dir, args.data_name+'_train{}.csv'.format(i))
                valid_data_path = os.path.join(args.data_dir, args.data_name+'_valid{}.csv'.format(i))
                logger.info("Reading {} and {}".format(train_data_path, valid_data_path))

                train_q_data, train_qa_data = data_loader.load_data(train_data_path)
                valid_q_data, valid_qa_data = data_loader.load_data(valid_data_path)

                auc, acc, loss = train(
                    model, 
                    train_q_data, train_qa_data, 
                    valid_q_data, valid_qa_data, 
                    result_log_path=result_csv_path,
                    args=args
                )

                aucs.append(auc)
                accs.append(acc)
                losses.append(loss)

    cross_validation_msg = "Cross Validation Result:\n"
    cross_validation_msg += "AUC: {:.2f} +/- {:.2f}\n".format(np.average(aucs)*100, np.std(aucs)*100)
    cross_validation_msg += "Accuracy: {:.2f} +/- {:.2f}\n".format(np.average(accs)*100, np.std(accs)*100)
    cross_validation_msg += "Loss: {:.2f} +/- {:.2f}\n".format(np.average(losses), np.std(losses))
    logger.info(cross_validation_msg)

    # write result
    result_msg = datetime.datetime.now().strftime("%Y-%m-%dT%H%M") + ','
    result_msg += str(args.dataset) + ','
    result_msg += str(args.memory_size) + ','
    result_msg += str(args.key_memory_state_dim) + ','
    result_msg += str(args.value_memory_state_dim) + ','
    result_msg += str(args.summary_vector_output_dim) + ','
    result_msg += str(np.average(aucs)*100) + ','
    result_msg += str(np.std(aucs)*100) + ','
    result_msg += str(np.average(accs)*100) + ','
    result_msg += str(np.std(accs)*100) + ','
    result_msg += str(np.average(losses)) + ','
    result_msg += str(np.std(losses)) + '\n'
    with open('results/all_result.csv', 'a') as f:
        f.write(result_msg)

if __name__=='__main__':
    cross_validation()