import argparse
import os
from shutil import copyfile

import paddle_torch as torch
import paddle_torch.nn as nn
import yaml
from svae_models import model_train
from utilis_svae import datasets

from zsl import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('template_path', type=str)
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    with open(args.template_path, 'r') as f:
        template = yaml.load(f)
    dataset_config = template['dataset'] 
    model_config = template['model']
    train_config = template['train']
    
    save_path = '{}_{}_{}'.format(dataset_config['save_path'],model_config['mid_size'], model_config['hidden_size'])
    # save_path = '{}_cls_{}'.format(dataset_config['save_path'],train_config['para'])

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    config_dir = os.path.join(save_path, 'config')
    log_dir = os.path.join(save_path, 'log')
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
        
    config_copy = '{}/{}'.format(config_dir, args.template_path)
    copyfile(args.template_path, config_copy)
      
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    '''    
    basis_config = template['basis']
    basis_dir = '{}/basis'.format(save_path)
    if not os.path.exists(basis_dir):
        os.mkdir(basis_dir)
    '''    
    if dataset_config['dataset_name'] == 'ImageNet':
        data = datasets.Imagenet(dataset_config['dataset_name'],
                            dataset_config['data_path'])  
    else:
     
        data = datasets.AwA2(dataset_config['dataset_name'],
                            dataset_config['data_path'])
    
                 
    dataset_setup_train = datasets.Dataset_setup(
                            data = data.train_set,
                            attrs = data.attrs ,
                            labels = data.train_labels 
                            )
        
    dataset_setup_val = datasets.Dataset_setup(
                            data = data.val_set,
                            attrs = data.attrs ,
                            labels = data.val_labels 
                            )
    dataset_setup_test_seen = datasets.Dataset_setup(
                            data = data.test_seen_set,
                            attrs = data.attrs ,
                            labels = data.test_seen_labels 
                            )
       
    dataset_setup_test_unseen = datasets.Dataset_setup(
                            data = data.test_unseen_set,
                            attrs = data.attrs ,
                            labels = data.test_unseen_labels 
                            )
    '''
    dataset_setup_test_seen = datasets.Dataset_setup_test(
                            data = data.test_seen_set,
                            attrs = data.attrs ,
                            labels = data.test_seen_labels,
                            append = data.test_seen_set 
                            )
       
    dataset_setup_test_unseen = datasets.Dataset_setup_test(
                            data = data.test_unseen_set,
                            attrs = data.attrs ,
                            labels = data.test_unseen_labels,
                            append = data.test_unseen_set_zsl
                            )
                            '''
    dataset_loader_train = torch.utils.data.DataLoader(dataset_setup_train, batch_size = train_config['batch_size'], shuffle= True, num_workers = 4)
    dataset_loader_val = torch.utils.data.DataLoader(dataset_setup_val, batch_size = train_config['batch_size'], shuffle= True, num_workers = 4)
    dataset_loader_test_seen = torch.utils.data.DataLoader(dataset_setup_test_seen, batch_size = train_config['batch_size'], shuffle= True, num_workers = 4)
    dataset_loader_test_unseen = torch.utils.data.DataLoader(dataset_setup_test_unseen, batch_size = train_config['batch_size'], shuffle= True, num_workers = 4)
    
   
    
    
    from svae_models import models
        
    attr_encoder = models.Attr_Encoder(model_config['attr_size'], model_config['mid_size'], model_config['hidden_size'])
    attr_decoder = models.Attr_Decoder(model_config['hidden_size'], model_config['mid_size'], model_config['attr_size'])
   
    encoder = models.Encoder(model_config['input_size'], model_config['mid_size'], model_config['hidden_size'])
    decoder = models.Decoder(model_config['hidden_size'], model_config['mid_size'], model_config['input_size'])

        
    from svae_models import model_train
    
    if dataset_config['GZSL']:
        classifier = models.LINEAR_LOGSOFTMAX(model_config['hidden_size'], model_config['classes'])
    else:
        classifier = models.LINEAR_LOGSOFTMAX(model_config['hidden_size'], data.unseen_labels.shape[0])
        
    zsl_classifier = models.LINEAR_LOGSOFTMAX(2048, data.unseen_labels.shape[0])
    
    # print(attr_encoder)
    # print(attr_decoder)
    # print(encoder)
    # print(decoder)
    # print(classifier)

    # zsl_classifier = Pre_train(data, dataset_config['dataset_name'])

   
    model_train_obj = model_train.Model_train(
                                  dataset_config['dataset_name'],
                                  encoder,
                                  decoder,
                                  attr_encoder,
                                  attr_decoder,
                                  classifier,
                                  dataset_loader_train,
                                  dataset_loader_test_unseen,
                                  dataset_loader_test_seen,
                                  criterion = nn.L1Loss(),
                                  lr = 1e-4,
                                  all_attrs = data.attrs,
                                  epoch = train_config['epoch'],
                                  save_path = save_path,
                                  save_every = train_config['save_every'],
                                  ifsample = model_config['ifsample'],
                                  data = data,
                                  GZSL = dataset_config['GZSL'],
                                  zsl_classifier = zsl_classifier
                                  )
    '''
    The following best epochs are obtained from the experiments of code implemented by PyTorch. 
    These can be references for the experiments of code implemented by PaddlePaddle.
    Dataset         AWA1      AWA2        CUB       SUN      FLO
    best_epoch      380       750     690,750,910  760,530  510,560,470
    '''
    
    epoch = 380

    # latent space visualization
    # model_train_obj.testing(epoch, sample_rate = 5)
    
    model_train_obj.draw_roc_curve(epoch, data)
    
    
    for i in range(epoch,2000,10):
        test_epoch = i
        threshold = model_train_obj.search_thres_by_traindata(test_epoch, dataset = data, n = 0.99)
        unseen_acc, _, ts, _ = model_train_obj.testing_2(test_epoch, test_class ='unseen', dataset = data, threshold = threshold)
        _, seen_acc,  tr, _ = model_train_obj.testing_2(test_epoch, test_class ='seen', dataset = data, threshold = threshold)
        print("epoch {}, unseen {}, seen {}, ts {}  tr {}, H {}".format(i, unseen_acc, seen_acc, ts, tr, 2*ts*tr/(ts + tr)))
        