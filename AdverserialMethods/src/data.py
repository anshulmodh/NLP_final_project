import numpy as np
import torch
import pdb

def get_duplicate_hypths(srcs):
  hyps_counts = {}
  for line in srcs:
    hyp = line.split("|||")[-1].strip()
    if hyp not in hyps_count:
      hyps_count[hyp] = 0
    hyps_count[hyp] += 1

  hyps_dup = set()
  for key,item in hyps_count.items():
    if item > 1:
      hyps_dup.add(item)
  return hyps_dup

def extract_from_file(lbls_file, srcs_file, max_sents, data_split, remove_dup):
  labels_to_int = None
  if "mpe" in lbls_file or "snli" in lbls_file or "multinli" in lbls_file or "sick" in lbls_file or "joci" in lbls_file or "glue" in lbls_file or "min" in lbls_file or "mnli" in lbls_file:
    labels_to_int = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
  elif "spr" in lbls_file or "dpr" in lbls_file or "fnplus" in lbls_file or "add_one" in lbls_file:
    labels_to_int = {'entailed': 0, 'not-entailed': 1}
  elif "scitail" in lbls_file:
    labels_to_int = {'entailment': 0, 'neutral': 1}
  else:
    print("Invalid lbls_file: %s" % (lbls_file))

  data = {'lbls': [], 'hypoths': [], 'premises': []}

  lbls = open(lbls_file).readlines()
  if "multinli" in srcs_file:
    srcs = open(srcs_file, 'rb').readlines()
  else:
    srcs = open(srcs_file).readlines()

  '''
  hyps_set = set()
  if remove_dup:
    hyps_set = get_duplicate_hypths(srcs)
  '''
  used_hyps = set()

  assert len(lbls) == len(srcs), "%s: %s labels and source files are not same length" % (lbls_file, data_split)
  num_duplicate_hyps = 0

  undecoded = 0

  added_sents = 0
  for i in range(len(lbls)):
    try:
      lbl = lbls[i].strip()
      text  = srcs[i]
      if "multinli" in srcs_file:
        text = text.decode(encoding='UTF-8')
      text = text.split('|||')
      hypoth = text[-1].strip()
      premise = text[0].strip()


      if remove_dup:
        if hypoth in used_hyps: 
          num_duplicate_hyps += 1
          continue

      if lbl not in labels_to_int:
        print("bad label: %s" % (lbl))
        continue

      if added_sents >= max_sents:
        continue

      data['lbls'].append(labels_to_int[lbl])
      data['hypoths'].append(hypoth)  
      data['premises'].append(premise)
      added_sents += 1

      if remove_dup:
        used_hyps.add(hypoth)
    except UnicodeDecodeError:
      undecoded += 1

  if remove_dup:
    print("Removed %d duplicate hypotheses out of %d total" % (num_duplicate_hyps, len(lbls)))
  print("Unable to decode", undecoded, "samples")

  return data

def get_nli_text(train_lbls_file, train_src_file, val_lbls_file, val_src_file, \
                   test_lbls_file, test_src_file, max_train_sents, max_val_sents, max_test_sents, remove_dup=False):
  labels = {}
  hypoths = {}

  train = extract_from_file(train_lbls_file, train_src_file, max_train_sents, "train", remove_dup)
  val = extract_from_file(val_lbls_file, val_src_file, max_val_sents, "val", remove_dup)
  test = extract_from_file(test_lbls_file, test_src_file, max_test_sents, "test", remove_dup)

  return train, val, test

def get_vocab(txt):
  vocab = set()
  for sent in txt:
    for word in sent.split():
      vocab.add(word)
  return vocab

# def get_word_vecs(vocab, embdsfile, lorelei_embds=False):
#   word_vecs = {}
#   with open(embdsfile, encoding="utf8") as f:
#     for line in f:
#       if line != "":
#         try:
#           word, vec = line.split(' ', 1)
#           if lorelei_embds:
#             word = word[4:]
#           if word in vocab:
#             word_vecs[word] = np.array(list(map(float, vec.split())))
#           else:
#             if word.lower() in vocab:
#               print(repr(word))
#         except Exception as e:
#           print(e)
#           print('--------->', repr(line))
#           1/0
#   print('Found {0}(/{1}) words with vectors'.format(
#             len(word_vecs), len(vocab)))
  # return word_vecs

def get_word_vecs(vocab, embdsfile, lorelei_embds=False):
  word_vecs = {}
  with open(embdsfile, encoding="utf8") as f:
    for line in f:
      word, vec = line.split(' ', 1)
      if lorelei_embds:
        word = word[4:]
      if word in vocab:
        word_vecs[word] = np.array(list(map(float, vec.split())))
      # elif word.lower() in vocab:
      #   word_vecs[word] = np.array(list(map(float, vec.split())))
  print('Found {0}(/{1}) words with vectors'.format(
            len(word_vecs), len(vocab)))
  if len(word_vecs) < 100:
    1/0
  return word_vecs
  
# def get_word_vecs(vocab, embdsfile, lorelei_embds=False):
#   word_vecs = {}
#   all_vecs = {}
#   with open(embdsfile, encoding="utf8") as f:
#     for line in f:
#       word, vec = line.split(' ', 1)
#       if lorelei_embds:
#         word = word[4:]
#       all_vecs[word] = vec
#   for word in vocab:
#     if word in all_vecs:
#       word_vecs[word] = np.array(list(map(float, all_vecs[word].split())))
#     else:
#       word = word.lower()
#       if word in all_vecs:
#         word_vecs[word] = np.array(list(map(float, all_vecs[word].split())))
#       # else:
#       #   print(word)
#       # elif word.lower() in vocab:
#       #   word_vecs[word] = np.array(list(map(float, vec.split())))
#   print('Found {0}(/{1}) words with vectors'.format(
#             len(word_vecs), len(vocab)))
#   if len(word_vecs) < 100:
#     1/0
#   return word_vecs



def build_vocab(txt, embdsfile, lorelei_embds=False):
  vocab = get_vocab(txt)
  vocab.add("OOV")
  word_vecs = get_word_vecs(vocab, embdsfile, lorelei_embds)
  print('Vocab size : {0}'.format(len(word_vecs)))
  return word_vecs 

def get_batch(batch, word_vec):
  # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
  lengths = np.array([len(x.split()) for x in batch])
  max_len = np.max(lengths)
  embed = np.zeros((max_len, len(batch), len(word_vec[next(iter(word_vec))])))

  for i in range(len(batch)):
    sent = batch[i].split()
    for j in range(len(batch[i].split())):
      if sent[j] not in word_vec:
        embed[j, i, :] = word_vec["OOV"]
      else:
        embed[j, i, :] = word_vec[sent[j]]

  return torch.from_numpy(embed).float(), lengths
