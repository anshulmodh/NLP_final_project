import os
import sys
import json
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli_text, build_vocab, get_batch
from models import SharedNLINet, SharedHypothNet, BLSTMEncoder
from mutils import get_optimizer

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

IDX2LBL = {}

def get_args():
  parser = argparse.ArgumentParser(description='Training NLI model based on just hypothesis sentence')

  # paths
  parser.add_argument("--embdfile", type=str, default='../data/embds/glove.840B.300d.txt', help="File containin the word embeddings")
  parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
  parser.add_argument("--model", type=str, help="Input model that has already been trained")
  parser.add_argument("--pred_file", type=str, default='preds', help="Suffix for the prediction files")
  parser.add_argument("--output_file", type=str, default='output', help="Suffix for the prediction files")
  parser.add_argument("--train_lbls_file", type=str, default='../data/snli_1.0/cl_snli_train_lbl_file', help="NLI train data labels file (SNLI or MultiNLI)")
  parser.add_argument("--train_src_file", type=str, default='../data/snli_1.0/cl_snli_train_source_file', help="NLI train data source file (SNLI or MultiNLI)")
  parser.add_argument("--val_lbls_file", type=str, default='../data/snli_1.0/cl_snli_val_lbl_file', help="NLI validation (dev) data labels file (SNLI or MultiNLI)")
  parser.add_argument("--val_src_file", type=str, default='../data/snli_1.0/cl_snli_val_source_file', help="NLI validation (dev) data source file (SNLI or MultiNLI)")
  parser.add_argument("--test_lbls_file", type=str, default='../data/snli_1.0/cl_snli_test_lbl_file', help="NLI test data labels file (SNLI or MultiNLI)")
  parser.add_argument("--test_src_file", type=str, default='../data/snli_1.0/cl_snli_test_source_file', help="NLI test data source file (SNLI or MultiNLI)")
  parser.add_argument("--dataset", type=str, default=None, help="What dataset to run on", choices=['snli', 'mnli', 'sick'],)


  # data
  parser.add_argument("--max_train_sents", type=int, default=10000000, help="Maximum number of training examples")
  parser.add_argument("--max_val_sents", type=int, default=10000000, help="Maximum number of validation/dev examples")
  parser.add_argument("--max_test_sents", type=int, default=10000000, help="Maximum number of test examples")
  parser.add_argument("--lorelei_embds", type=int, default=0, help="Whether to use multilingual embeddings released for LORELEI. This requires cleaning up words since wordsare are prefixed with the language. 0 for no, 1 for yes (Default is 0)")

  # model
  parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
  parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
  parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
  parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
  parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
  parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
  parser.add_argument("--batch_size", type=int, default=256)
  parser.add_argument("--hypoth_only", type=bool, default=False, help="h model")

  # gpu
  parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
  parser.add_argument("--seed", type=int, default=1234, help="seed")


  #misc
  parser.add_argument("--verbose", type=int, default=1, help="Verbose output")

  params, _ = parser.parse_known_args()

  # print parameters passed, and all parameters
  print('\ntogrep : {0}\n'.format(sys.argv[1:]))
  # print(params)

  return params

def get_data(args):
  if args.dataset is None or args.dataset == 'snli':
    return args
  if args.dataset == 'mnli':
    args.train_lbls_file = "../data/multinli_1.0/cl_multinli_train_lbl_file"
    args.train_src_file = "../data/multinli_1.0/cl_multinli_train_source_file"
    args.val_lbls_file = "../data/multinli_1.0/cl_multinli_dev_matched_lbl_file"
    args.val_src_file = "../data/multinli_1.0/cl_multinli_dev_matched_source_file"
    args.test_lbls_file = "../data/multinli_1.0/cl_multinli_dev_mismatched_lbl_file"
    args.test_src_file = "../data/multinli_1.0/cl_multinli_dev_mismatched_source_file"
  if args.dataset == 'sick':
    args.train_lbls_file = "../data/sick/cl_sick_train_lbl_file"
    args.train_src_file = "../data/sick/cl_sick_train_source_file"
    args.val_lbls_file = "../data/sick/cl_sick_val_lbl_file"
    args.val_src_file = "../data/sick/cl_sick_val_source_file"
    args.test_lbls_file = "../data/sick/cl_sick_test_lbl_file"
    args.test_src_file = "../data/sick/cl_sick_test_source_file"
  return args

def compute_stats(preds, actuals):
    confusion = confusion_matrix(actuals, preds)
    print('Confusion Matrix\n')
    print(confusion)

    print('\nClassification Report\n')
    report = classification_report(actuals, preds, target_names=['entailment', 'neutral', 'contradiction'])
    print(report)

    return confusion, report

def output_metrics(output_file, acc, preds, actuals):
  confusion, report = compute_stats(preds, actuals)
  data = {
      'sk_accuracy': accuracy_score(actuals, preds),
      'torch_accuracy': acc,
      'confusion': confusion.tolist(),
      'classsification_report': report
  }
  f = open(output_file, 'w')
  f.write(json.dumps(data))
  f.close()

def evaluate(epoch, valid, params, word_vec, model, eval_type, filepath):
  pred_file = filepath + args.pred_file
  model.eval()
  correct = 0.
  global val_acc_best, lr, stop_training, adam_stop

  #if eval_type == 'valid':
  print('\n{0} : Epoch {1}'.format(eval_type, epoch))

  hypoths = valid['hypoths'] #if eval_type == 'valid' else test['s1']
  premises = valid['premises'] #if eval_type == 'valid' else test['s2']
  target = valid['lbls']

  out_preds_f = open(pred_file, "w")
  preds = []
  actuals = []

  for i in range(0, len(hypoths), params.batch_size):
    # prepare batch
    hypoths_batch, hypoths_len = get_batch(hypoths[i:i + params.batch_size], word_vec)
    premises_batch, premises_len = get_batch(premises[i:i + params.batch_size], word_vec)
    tgt_batch = None
    if params.gpu_id > -1:
      hypoths_batch = Variable(hypoths_batch.cuda())
      premises_batch = Variable(premises_batch.cuda())
      tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()
    else:
      hypoths_batch = Variable(hypoths_batch)
      premises_batch = Variable(premises_batch)
      tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size]))

    # model forward
    if args.hypoth_only:
      output = model((hypoths_batch, hypoths_len))
    else:
      output = model((premises_batch, premises_len), (hypoths_batch, hypoths_len))
    all_preds = output.data.max(1)[1]
    for pred in all_preds:
      out_preds_f.write(IDX2LBL[pred.item()] + "\n")
    correct += all_preds.long().eq(tgt_batch.data.long()).cpu().sum()

    preds += all_preds.tolist()
    actuals += tgt_batch.tolist()
  

  out_preds_f.close()
  # save model
  eval_acc = round((100.0 * correct / len(hypoths)).item(), 2)
  print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))

  output_metrics(filepath + args.output_file, eval_acc, preds, actuals)

  return eval_acc

def main(args):
  print("main")

  """
  SEED
  """
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  """
  DATA
  """
  args = get_data(args)

  train, val, test = get_nli_text(args.train_lbls_file, args.train_src_file, args.val_lbls_file, \
                                    args.val_src_file, args.test_lbls_file, args.test_src_file, \
                                    args.max_train_sents, args.max_val_sents, args.max_test_sents)

  # word_vecs = build_vocab(val['hypoths'] + val['premises'], \
  #                         args.embdfile, args.lorelei_embds)

  word_vecs = build_vocab(train['hypoths'] + val['hypoths'] + test['hypoths'] + train['premises'] + val['premises'] + test['premises'], \
                          args.embdfile, args.lorelei_embds)
  args.word_emb_dim = len(word_vecs[next(iter(word_vecs))])

  lbls_file = args.train_lbls_file
  global IDX2LBL
  if "mpe" in lbls_file or "snli" in lbls_file or "multinli" in lbls_file or "sick" in lbls_file or "joci" in lbls_file or "glue" in lbls_file or "mnli" in lbls_file:
    IDX2LBL = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
  elif "spr" in lbls_file or "dpr" in lbls_file or "fnplus" in lbls_file or "add_one" in lbls_file:
    IDX2LBL = {0: 'entailed', 1: 'not-entailed', 2: 'not-entailed'}
  elif "scitail" in lbls_file:
    IDX2LBL = {0: 'entailment', 1: 'neutral', 2: 'neutral'}

  model = torch.load(args.model)
  print(model)

  # loss
  weight = torch.FloatTensor(args.n_classes).fill_(1)
  loss_fn = nn.CrossEntropyLoss(weight=weight)
  loss_fn.size_average = False

  if args.gpu_id > -1:
    model.cuda()
    loss_fn.cuda()

  """
  Train model on Natural Language Inference task
  """
  epoch = 1

  if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)

  # for pair in [(val, 'val')]:
  # for pair in [(val, 'val'), (test, 'test')]:
  for pair in [(train, 'train'), (val, 'val'), (test, 'test')]:
    #args.batch_size = len(pair[0]['lbls'])
    print("-" * 30)
    eval_acc = evaluate(0, pair[0], args, word_vecs, model, pair[1], "%s/%s_" % (args.outputdir, pair[1]))
    #epoch, valid, params, word_vec, nli_net, eval_type


if __name__ == '__main__':
  args = get_args()
  main(args)
