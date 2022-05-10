import itertools
import os

job_string_base = '''
#!/bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -l gpu 

source activate transformer
'''

# All parameters that need to be set.
outpath_base_m1 = '/robust-nli-fixed/src/new_method1/'
outpath_base_m2 = '/robust-nli-fixed/src/new_method2/'
glove_path = "/robust-nli-fixed/data/embds/glove.840B.300d.txt"
# snli_path = "/idiap/temp/rkarimi/datasets/SNLI"


def submit_job(curr_job, filename, outpath_base):
    job_name = "template.job"
    with open(job_name, "w") as f:
       f.write(curr_job)
    os.system("qsub -V -N {0} -e {1}.err -o {1}.out template.job".format(filename, os.path.join(outpath_base, filename)))


# Hyper-parameter tuning for method 1.

# if not os.path.exists("method1_baseline"):
#     os.makedirs("method1_baseline")
# job_string = '''python train.py --embdfile {3} --outputdir {0}/alpha_{1}_beta_{2}  --pool_type max --n_classes 3  --adv_lambda {2} --adv_hyp_encoder_lambda {1} --nli_net_adv_hyp_encoder_lambda 0 --random_premise_frac 0 --enc_lstm_dim 512''' 
# # alphas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
# # betas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
# alphas = [0.2, 0.05, 0.1, 1]
# betas = [0.1, 1, 0.4, 0.1]
# # trainingfiles = ["min0.6,0.1-0.5.txt", "min0.6,0.2-0.8.txt", "min0.8,0.1-0.5.txt", "min0.8,0.2-0.8.txt", "min0.8,0.5-0.8.txt", "min0.6,0.5-0.8.txt"]
# # alphas = [0.2]
# # betas = [0.1]
# for j in range(len(alphas)):
#     alpha, beta = alphas[j], betas[j]  
#     curr_job = job_string.format(outpath_base_m1, alpha, beta, glove_path)
#     os.system(curr_job)
#     #    submit_job(curr_job, "m1"+str(alpha)+str(beta), outpath_base)


# # Hyper-parameter tuning for method 2
# if not os.path.exists("method2_baseline"):
#     os.makedirs("method2_baseline")
# job_string = '''python train.py --embdfile {3} --outputdir {0}/alpha_{1}_beta_{2}  --pool_type max --n_classes 3 --adv_lambda 0 --adv_hyp_encoder_lambda 0 --nli_net_adv_hyp_encoder_lambda {2} --random_premise_frac {1} --enc_lstm_dim 512''' 
# # alphas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
# # betas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
# # trainingfiles = ["min0.6,0.1-0.5.txt", "min0.6,0.2-0.8.txt", "min0.8,0.1-0.5.txt", "min0.8,0.2-0.8.txt", "min0.8,0.5-0.8.txt", "min0.6,0.5-0.8.txt"]
# alphas = [0.05, 0.1]
# betas = [0.05, 0.05]
# for j in range(len(alphas)):
#     alpha, beta = alphas[j], betas[j]  
#     curr_job = job_string.format(outpath_base_m2, alpha, beta, glove_path)
#     os.system(curr_job)
#     #    submit_job(curr_job, "m2"+str(alpha)+str(beta), outpath_base)


if not os.path.exists(outpath_base_m1):
    os.makedirs(outpath_base_m1)
job_string = '''python train.py --embdfile {3} --outputdir {0}/alpha_{1}_beta_{2}_{5}  --pool_type max --n_classes 3  --adv_lambda {2} --adv_hyp_encoder_lambda {1} --nli_net_adv_hyp_encoder_lambda 0 --random_premise_frac 0 --enc_lstm_dim 512 --train_src_file {4} --train_lbls_file {6}''' 
# alphas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
# betas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
alphas = [1]
betas = [0.1]
# trainingfiles = ["min0.6,0.1-0.5.txt", "min0.6,0.2-0.8.txt", "min0.8,0.1-0.5.txt", "min0.8,0.2-0.8.txt", "min0.8,0.5-0.8.txt", "min0.6,0.5-0.8.txt"]
trainingfiles = ["min0.6,scaled--fixed.txt"]
labels = ["../data/snli_1.0/cl_snli_train_lbl_file"]
alphas = [0.1, 1]
betas = [0.4, 0.1]
for i in range(len(trainingfiles)):
    f, l, = trainingfiles[i], labels[i], 
    for j in range(len(alphas)):
        alpha, beta = alphas[j], betas[j]  
        curr_job = job_string.format(outpath_base_m1, alpha, beta, glove_path, f, f[:f.find(".txt")], l)
        os.system(curr_job)
    #    submit_job(curr_job, "m1"+str(alpha)+str(beta), outpath_base)


if not os.path.exists(outpath_base_m1):
    os.makedirs(outpath_base_m1)
job_string = '''python train.py --embdfile {3} --outputdir {0}/alpha_{1}_beta_{2}_{5}  --pool_type max --n_classes 3  --adv_lambda {2} --adv_hyp_encoder_lambda {1} --nli_net_adv_hyp_encoder_lambda 0 --random_premise_frac 0 --enc_lstm_dim 512 --train_src_file {4} --train_lbls_file {6}''' 
# alphas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
# betas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
alphas = [0.2, 0.05, 0.1, 1]
betas = [0.1, 1, 0.4, 0.1]
# trainingfiles = ["min0.6,0.1-0.5.txt", "min0.6,0.2-0.8.txt", "min0.8,0.1-0.5.txt", "min0.8,0.2-0.8.txt", "min0.8,0.5-0.8.txt", "min0.6,0.5-0.8.txt"]
trainingfiles = ["min0.7,scaled--samples.txt", "min0.9,scaled--samples.txt"]
labels = ["min0.7,scaled--samples_labels.txt", "min0.9,scaled--samples_labels.txt"]
# alphas = [0.2]
# betas = [0.1]
for i in range(len(trainingfiles)):
    f, l, = trainingfiles[i], labels[i], 
    for j in range(len(alphas)):
        alpha, beta = alphas[j], betas[j]  
        curr_job = job_string.format(outpath_base_m1, alpha, beta, glove_path, f, f[:f.find(".txt")], l)
        os.system(curr_job)
    #    submit_job(curr_job, "m1"+str(alpha)+str(beta), outpath_base)


# Hyper-parameter tuning for method 2
if not os.path.exists(outpath_base_m2):
    os.makedirs(outpath_base_m2)
job_string = '''python train.py --embdfile {3} --outputdir {0}/alpha_{1}_beta_{2}_{5}  --pool_type max --n_classes 3 --adv_lambda 0 --adv_hyp_encoder_lambda 0 --nli_net_adv_hyp_encoder_lambda {2} --random_premise_frac {1} --enc_lstm_dim 512 --train_src_file {4} --train_lbls_file {6}''' 
# alphas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
# betas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
# trainingfiles = ["min0.6,0.1-0.5.txt", "min0.6,0.2-0.8.txt", "min0.8,0.1-0.5.txt", "min0.8,0.2-0.8.txt", "min0.8,0.5-0.8.txt", "min0.6,0.5-0.8.txt"]
trainingfiles = ["min0.8,scaled--samples.txt", "min0.6,scaled--samples.txt", "min0.6,scaled--fixed.txt", "min0.8,scaled--fixed.txt", "min0.7,scaled--fixed.txt", "min0.9,scaled--fixed.txt", "min0.7,scaled--samples.txt", "min0.9,scaled--samples.txt"]
labels = ["min0.8,scaled--samples_labels.txt", "min0.6,scaled--samples_labels.txt", "../data/snli_1.0/cl_snli_train_lbl_file", "../data/snli_1.0/cl_snli_train_lbl_file",  "../data/snli_1.0/cl_snli_train_lbl_file",  "../data/snli_1.0/cl_snli_train_lbl_file", "min0.7,scaled--samples_labels.txt", "min0.9,scaled--samples_labels.txt"]
alphas = [0.05, 0.1]
betas = [0.05, 0.05]
for i in range(len(trainingfiles)):
    f, l, = trainingfiles[i], labels[i], 
    for j in range(len(alphas)):
        alpha, beta = alphas[j], betas[j]  
        curr_job = job_string.format(outpath_base_m2, alpha, beta, glove_path, f, f[:f.find(".txt")], l)
        os.system(curr_job)
    #    submit_job(curr_job, "m2"+str(alpha)+str(beta), outpath_base)
