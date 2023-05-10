import os
import pickle

import ipdb
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import paddle
import paddle_torch as torch
import paddle_torch.nn as nn
import paddle_torch.nn.functional as F
import pandas as pd
from hyperspherical_vae.distributions import VonMisesFisher
from paddle_torch.tensor import convertTensor
from utilis_svae import emd


def norm_data(visual_features):
    for i in range(visual_features.shape[0]):
        visual_features[i,:] = visual_features[i,:]/np.linalg.norm(visual_features[i,:]) 
    return visual_features
    
    
class Model_train(object):
    def __init__(self, 
                 dataset_name,
                 encoder,
                 decoder,
                 attr_encoder,
                 attr_decoder,
                 classifier,
                 train_loader,
                 test_loader_unseen,
                 test_loader_seen,
                 criterion,
                 lr = 1e-3,
                 all_attrs = None,
                 epoch = 10000,
                 save_path = "/data/xingyu/wae_lle/experiments/",
                 save_every = 1,
                 iftest = False,
                 ifsample = False,
                 data = None,
                 GZSL = True,
                 zsl_classifier = None
                 ):  
        self.dataset_name = dataset_name
        self.encoder = encoder
        self.decoder = decoder
        self.attr_encoder = attr_encoder
        self.attr_decoder = attr_decoder
        self.classifier = classifier
        self.zsl_classifier = zsl_classifier
        self.train_loader = train_loader
        self.test_loader_unseen = test_loader_unseen
        self.test_loader_seen = test_loader_seen
           
        self.criterion = criterion
        self.crossEntropy_Loss = nn.NLLLoss()
        
        self.all_attrs = all_attrs
        self.lr = lr
        self.epoch = epoch
        self.save_path = save_path
        self.save_every = save_every
        self.ifsample = ifsample
        self.data = data
        self.GZSL = GZSL
        self.distribution = 'vmf'
        self.sinkhorn = emd.SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
        
        if iftest:
            log_dir = '{}/log'.format(self.save_path)
            #general.logger_setup(log_dir, 'results__')
        
    def save_checkpoint(self,state, filename = 'checkpoint.pdparams.tar'):
        torch.save(state, filename)  
         
    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
        else:
            raise NotImplemented

        return q_z
        
    def compute_acc(self,trues, preds):
        """
        Given true and predicted labels, computes average class-based accuracy.
        """

        # class labels in ground-truth samples
        classes = np.unique(trues)
        # class-based accuracies
        cb_accs = np.zeros(classes.shape, np.float32)
        #ipdb.set_trace()
        for i, label in enumerate(classes):
            inds_ci = np.where(trues == label)[0]

            cb_accs[i] = np.mean(
              np.equal(
              trues[inds_ci],
              preds[inds_ci]
            ).astype(np.float32)
        )
        #ipdb.set_trace()
        return np.mean(cb_accs)   
      
    def training(self, checkpoint = -1):
        log_dir = '{}/log'.format(self.save_path)
        #general.logger_setup(log_dir)
        ww=0.1
        cl=1.0
        cr=1.0
    
        if checkpoint > 0:
            file_encoder = 'Checkpoint_{}_Enc.pdparams.tar'.format(checkpoint)
            file_decoder = 'Checkpoint_{}_Dec.pdparams.tar'.format(checkpoint)
            file_attr_encoder = 'Checkpoint_{}_attr_Enc.pdparams.tar'.format(checkpoint)
            file_attr_decoder = 'Checkpoint_{}_attr_Dec.pdparams.tar'.format(checkpoint)
            file_classifier = 'Checkpoint_{}_classifier.pdparams.tar'.format(checkpoint)
                
            enc_path = os.path.join(self.save_path, file_encoder)
            dec_path = os.path.join(self.save_path, file_decoder)
            attr_enc_path = os.path.join(self.save_path, file_attr_encoder)
            attr_dec_path = os.path.join(self.save_path, file_attr_decoder)
            classifier_path = os.path.join(self.save_path, file_classifier)
                
            enc_checkpoint = paddle.load(enc_path)
            self.encoder.set_state_dict(enc_checkpoint['state_dict'])
        
            dec_checkpoint = paddle.load(dec_path)
            self.decoder.set_state_dict(dec_checkpoint['state_dict'])
            
            attr_enc_checkpoint = paddle.load(attr_enc_path)
            self.attr_encoder.set_state_dict(attr_enc_checkpoint['state_dict'])
            
            attr_dec_checkpoint = paddle.load(attr_dec_path)
            self.attr_decoder.set_state_dict(attr_dec_checkpoint['state_dict'])
            
            classifier_checkpoint = paddle.load(classifier_path)
            self.classifier.set_state_dict(classifier_checkpoint['state_dict'])
                
        self.encoder.train()
        self.decoder.train()
        self.attr_encoder.train() 
        self.attr_decoder.train()
        self.classifier.train()
        
        enc_optim = paddle.optimizer.Adam(parameters=self.encoder.parameters(), learning_rate=self.lr)
        dec_optim = paddle.optimizer.Adam(parameters=self.decoder.parameters(), learning_rate=self.lr)
        attr_enc_optim = paddle.optimizer.Adam(parameters=self.attr_encoder.parameters(), learning_rate=self.lr)
        attr_dec_optim = paddle.optimizer.Adam(parameters=self.attr_decoder.parameters(), learning_rate=self.lr)
        classifier_optim = paddle.optimizer.Adam(parameters=self.classifier.parameters(), learning_rate=self.lr)
            
        self.criterion = paddle.nn.L1Loss()
        self.crossEntropy_Loss = paddle.nn.NLLLoss()

        for epoch in range(checkpoint+1, self.epoch):
            step = 0            
            train_data_iter = iter(self.train_loader)
            for i_batch, sample_batched in enumerate(self.train_loader):                      
                input_data = sample_batched['feature']
                input_label = sample_batched['label']
                input_attr = sample_batched['attr']
                
                batch_size = input_data.shape[0]

                input_data = input_data.astype('float32')
                input_label = input_label.astype('int64').flatten(0, -1)
                input_attr = input_attr.astype('float32').squeeze()
                all_attrs = paddle.to_tensor(self.all_attrs).astype('float32')
                        
                self.encoder.clear_gradients()
                self.decoder.clear_gradients()
                self.attr_encoder.clear_gradients()
                self.attr_decoder.clear_gradients()
                self.classifier.clear_gradients()
                
                m1, s1 = self.encoder(input_data)
                z1 = self.reparameterize(m1, s1)

                m2, s2 = self.attr_encoder(all_attrs)
                z2 = self.reparameterize(m2, s2)
                
                z_x = z1.rsample()
                z_attr = z2.rsample()[input_label]
                
                sub_batch_size = 10
                z_x_2 = z1.rsample(sub_batch_size).transpose(perm=[1,0,2])
                z_attr_2 = z2.rsample(sub_batch_size).transpose(perm=[1,0,2])[input_label]
                
            
                z_input = paddle.concat((z_attr.squeeze(), z_x), axis=0) 
                label_input = paddle.concat((input_label, input_label), axis=0)
             
                cls_out = self.classifier(z_input)
                cls_loss = self.crossEntropy_Loss(cls_out, label_input)
                
                
                # Used for ablation experiments
                '''
                x_recon = self.decoder(z_x)
                recon_loss = self.criterion(x_recon, input_data)
                attr_recon = self.attr_decoder(z_attr)
                attr_loss = self.criterion(attr_recon, input_attr)
             
                x_recon_cr = self.decoder(z_attr)
                recon_loss_cr = self.criterion(x_recon_cr, input_data)
                attr_recon_cr = self.attr_decoder(z_x)
                attr_loss_cr = self.criterion(attr_recon_cr, input_attr)
                cr_loss = recon_loss_cr + attr_loss_cr
                '''
                #original code
                x_recon = self.decoder(z_input)
                recon_loss = self.criterion(x_recon, paddle.concat((input_data,input_data), axis=0))

                attr_fake = self.attr_decoder(z_input)
                attr_loss = self.criterion(attr_fake, paddle.concat((input_attr,input_attr), axis=0))

                # x_recon = self.decoder(z_x)
                # recon_loss = self.criterion(x_recon, input_data)
                # attr_recon = self.attr_decoder(z_attr)
                # attr_loss = self.criterion(attr_recon, input_attr)
                
     
                dist, P, C = self.sinkhorn(z_x_2, z_attr_2)
                #ipdb.set_trace()
            
                KL_loss = dist.mean()
               
                total_loss =  recon_loss *cr + KL_loss * ww  + attr_loss *cr + cls_loss* cl  
            
                total_loss.backward()
            
                enc_optim.step()
                dec_optim.step()
                attr_enc_optim.step()
                attr_dec_optim.step()
                classifier_optim.step()

                self.attr_encoder.clear_gradients()
                self.attr_decoder.clear_gradients()
                self.classifier.clear_gradients()

                m2, s2 = self.attr_encoder(all_attrs)
                zz = self.reparameterize(m2, s2)
                
                z2 = zz.rsample()
                attr_fake = self.attr_decoder(z2)
                attr_loss = self.criterion(attr_fake, all_attrs)
                # z_attr_unseen = z2[self.data.unseen_mask,:]
                z_attr_unseen = z2[self.data.unseen_mask]
                # z_attr_seen = z2[self.data.seen_mask,:]
                ####
                # x_fake = self.decoder(z_attr_seen)
                # m3,s3 = self.encoder(x_fake)
                # z3 = self.reparameterize(m3, s3)
                # z_x = z3.rsample()
                # cross_cycle_loss = self.criterion(z_x, z_attr_seen)

                x_fake = self.decoder(z_attr_unseen)
                m3, s3 = self.encoder(x_fake)
                z3 = self.reparameterize(m3, s3)

                sub_batch_size = 10
                z_x_2 = z3.rsample(sub_batch_size).transpose(perm=[1,0,2])
                z_attr_2 = zz.rsample(sub_batch_size).transpose(perm=[1,0,2])[self.data.unseen_mask]
                
                dist, P, C = self.sinkhorn(z_x_2, z_attr_2)

                kl = dist.mean()

                cls_out = self.classifier(z2)
                label_all = paddle.to_tensor(self.data.label_all).astype('int64').flatten(0, -1)
                cls_loss = self.crossEntropy_Loss(cls_out, label_all)

                # cross_cycle_loss_unseen = self.criterion(z_x, z_attr_unseen)
                # ####
                # attr_fake = self.attr_decoder(z_attr)
                # attr_loss = self.criterion(attr_fake, all_attrs)
                # attr_loss.backward()
                # all_loss = attr_loss*cr+cl*cls_loss+w*kl
                all_loss = attr_loss * cr + cl * cls_loss + ww * kl
                all_loss.backward()

                attr_enc_optim.step()
                attr_dec_optim.step()
                classifier_optim.step()
                #
                # enc_optim.step()
                # dec_optim.step()
                
                step += 1
            
                if (step + 1) % 50 == 0:
                    print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f KL_Loss: %.4f, attr_Recon Loss: %.4f, cls_Loss: %.4f, k1: %.4f, k2: %.4f, u: %.4f" %
                          (epoch, self.epoch, step , len(self.train_loader), recon_loss.item(), KL_loss.item(), attr_loss.item(), cls_loss.item(), s1.mean().item(), s2.mean().item(), paddle.dot(z_x[1,:], z_attr.squeeze()[1,:]).item()))
                    ######
                    # s = "Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f KL_Loss: %.4f, attr_Recon Loss: %.4f, cls_Loss: %.4f, k1: %.4f, k2: %.4f, u: %.4f" % (
                    #     epoch, self.epoch, step , len(self.train_loader), recon_loss.item(), KL_loss.item(), attr_loss.item(), cls_loss.item(), s1.mean().item(), s2.mean().item(), paddle.dot(z_x[1,:], z_attr.squeeze()[1,:]).item())
                    # print(s)
                    # with open(log_dir+'/outlog.txt', 'a') as fp:
                    #     fp.write(s+'\n')
                    #     fp.close()

            if epoch % self.save_every ==0: 
            
                file_encoder = 'Checkpoint_{}_Enc.pdparams.tar'.format(epoch)
                file_decoder = 'Checkpoint_{}_Dec.pdparams.tar'.format(epoch)
                file_attr_encoder = 'Checkpoint_{}_attr_Enc.pdparams.tar'.format(epoch)
                file_attr_decoder = 'Checkpoint_{}_attr_Dec.pdparams.tar'.format(epoch)
                file_classifier = 'Checkpoint_{}_classifier.pdparams.tar'.format(epoch)
             
                file_name_enc = os.path.join(self.save_path, file_encoder)
                file_name_dec = os.path.join(self.save_path, file_decoder)
                file_name_attr_enc = os.path.join(self.save_path, file_attr_encoder)
                file_name_attr_dec = os.path.join(self.save_path, file_attr_decoder)
                file_name_classifier = os.path.join(self.save_path, file_classifier)
                
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.encoder.state_dict(), 
                     'optimizer': enc_optim.state_dict()}, 
                     file_name_enc)
                                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.decoder.state_dict(), 
                     'optimizer': dec_optim.state_dict()}, 
                     file_name_dec)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_encoder.state_dict(), 
                     'optimizer': attr_enc_optim.state_dict()}, 
                     file_name_attr_enc)
                     
                self.save_checkpoint(
                    {'epoch':epoch, 
                     'state_dict': self.attr_decoder.state_dict(), 
                     'optimizer': attr_dec_optim.state_dict()}, 
                     file_name_attr_dec)   
                self.save_checkpoint(
                    {'epoch':epoch,
                     'state_dict': self.classifier.state_dict(), 
                     'optimizer': classifier_optim.state_dict()}, 
                     file_name_classifier)   
            
    # def search_thres_by_sample(self, attrs, n = 10000):
    #     min_thres = 100
    #     m, s = self.attr_encoder(attrs)
      
    #     z = []
    #     for i in range(n):
    #         z_fake = self.reparameterize(m, s).rsample()
    #         dist = F.cosine_similarity(m, z_fake)
    #         z.append(z_fake)
    #         thres = dist.min()
    #         if min_thres > thres:
    #             min_thres = thres
        
    #     return min_thres
        
    def load_models(self, epoch):
        file_encoder = 'Checkpoint_{}_Enc.pdparams.tar'.format(epoch)
        file_decoder = 'Checkpoint_{}_Dec.pdparams.tar'.format(epoch)
        file_attr_encoder = 'Checkpoint_{}_attr_Enc.pdparams.tar'.format(epoch)  
        file_classifier = 'Checkpoint_{}_classifier.pdparams.tar'.format(epoch)  
        enc_path = os.path.join(self.save_path, file_encoder)
        dec_path = os.path.join(self.save_path, file_decoder)
        attr_enc_path = os.path.join(self.save_path, file_attr_encoder)
        classifier_path = os.path.join(self.save_path, file_classifier)

        enc_checkpoint = torch.load(enc_path)
        self.encoder.set_state_dict(enc_checkpoint['state_dict'])
        dec_checkpoint = torch.load(dec_path)
        self.decoder.set_state_dict(dec_checkpoint['state_dict'])
        attr_enc_checkpoint = torch.load(attr_enc_path)
        self.attr_encoder.set_state_dict(attr_enc_checkpoint['state_dict'])
        classifier_checkpoint = torch.load(classifier_path)
        self.classifier.set_state_dict(classifier_checkpoint['state_dict'])       
        
        # Load the ZSL classifiers. These ZSL classifiers can be replaced by any SOTA models! 
        if self.dataset_name == 'AWA1':
            zsl_classifier_checkpoint = torch.load("../zsl_models/awa1_Checkpoint_24_Classifier.pdparams.tar")
        elif self.dataset_name == 'AWA2':
            zsl_classifier_checkpoint = torch.load("../zsl_models/awa2_Checkpoint_9_Classifier.pdparams.tar")
        elif self.dataset_name == 'CUB':
            zsl_classifier_checkpoint = torch.load("../zsl_models/cub_Checkpoint_7_Classifier.pdparams.tar")
        elif self.dataset_name == 'FLO':
            zsl_classifier_checkpoint = torch.load("../zsl_models/flo_Checkpoint_24_Classifier.pdparams.tar")
        elif self.dataset_name == 'SUN':
            zsl_classifier_checkpoint = torch.load("../zsl_models/sun_Checkpoint_14_Classifier.pdparams.tar")
        
        self.zsl_classifier.set_state_dict(zsl_classifier_checkpoint['state_dict'])
        
        self.encoder.eval()
        self.decoder.eval()
        self.attr_encoder.eval()
        self.zsl_classifier.eval()
        # self.zsl_classifier.model.eval()
        
    def search_thres_by_traindata(self, epoch, dataset = None, n = 0.95):
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = dataset.seen_labels
        unseen_labels = dataset.unseen_labels
        self.load_models(epoch)

        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []    
        all_anchors = self.attr_encoder(all_attrs)[0]
        all_anchors = convertTensor(all_anchors)
        seen_idx = seen_labels - 1
        unseen_idx = unseen_labels -1
        
        seen_anchors = all_anchors[seen_idx.tolist(),:]
        unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        seen_count = 0
        seen_all = 0
        unseen_count = 0
        unseen_all = 0
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        
        for i_batch, sample_batched in enumerate(self.train_loader):
            input_data = sample_batched['feature']
            input_label = sample_batched['label']   
            input_attr = sample_batched['attr']

            ######
            input_data = convertTensor(input_data)
            input_label = convertTensor(input_label)
            input_attr = convertTensor(input_attr)

            batch_size = input_data.size()[0]

            input_data = input_data.astype('float32')
            input_label = input_label.astype('int64')  
            input_attr = input_attr.astype('float32')
                                
            m, s = self.encoder(input_data)
            m = convertTensor(m)
            s = convertTensor(s)
            #z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                kk = input_label[k,:]+1
                z_tile = z_real[k,:].repeat(seen_anchors.shape[0]).view(seen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, seen_anchors)
                if min_thres>dist.max():
                    min_thres = dist.max()
                mean_dist += dist.max()
                dist_list.append(dist.max().item())
            
        dist_array = np.array(dist_list)
        idx = dist_array.shape[0] * (1.0 - n)
        thres  = np.sort(dist_array)[int(idx)]
      
        return thres 

    #    def testing_1(self, epoch, test_class = 'seen', dataset = None, D = None, threshold = 0.99):
    #     coeff_fun_map = {
    #         'optimized_seq':
    #         lambda data, dic: cpp.decomp_simplex_sequence(
    #             data, dic, n_smooth_iter=1, sub_window_size=3, lambda1=0.1),
    #         'optimized': cpp.decomp_simplex
    #     }
    #     coeff_fun = coeff_fun_map['optimized']
    #     if test_class == 'seen':
    #         test_loader = self.test_loader_seen
    #     elif test_class == 'unseen':
    #         test_loader = self.test_loader_unseen
    #
    #     all_attrs = torch.Tensor(dataset.attrs).float().cuda()
    #     seen_labels = dataset.seen_labels
    #     unseen_labels = dataset.unseen_labels
        
    #     if isinstance(threshold, np.ndarray):
    #         thresholds = threshold
    #     else:
    #         thresholds = np.ones(seen_labels.shape[0]) * threshold
            
    #     self.load_models(epoch) 
    #     z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
    #     all_anchors = self.attr_encoder(all_attrs)[0]
    #     all_anchors = convertTensor(all_anchors)      
    #     seen_idx = seen_labels - 1
    #     unseen_idx = unseen_labels -1
        
    #     seen_anchors = all_anchors[seen_idx.tolist(),:]
    #     unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        
    #     seen_count = 0
    #     seen_all = 1
    #     unseen_count = 0
    #     unseen_all = 1
    #     all_count = 0
    #     min_thres = 10
    #     mean_dist = 0
    #     dist_list = []
    #     pred = []
    #     gt = []
    #     for i_batch, sample_batched in enumerate(test_loader):
    #         input_data = sample_batched['feature']
    #         input_label = sample_batched['label']   
    #         input_attr = sample_batched['attr']
    #         
    #         ######
    #         input_data = convertTensor(input_data)
    #         input_label = convertTensor(input_label)
    #         input_attr = convertTensor(input_attr)
    #
    #         batch_size = input_data.size()[0]           
    #
    #         input_data = input_data.astype('float32')
    #         input_label = input_label.astype('int64')
    #         input_attr = input_attr.astype('float32')
                                
    #         m, s = self.encoder(input_data)
    #         m = convertTensor(m)
    #         s = convertTensor(s) 
    #         z_real = self.reparameterize(m, s).rsample().squeeze()
    #         z_real = m.squeeze()
            
    #         for k in range(z_real.shape[0]):
    #             input_k = input_data[k,:]
    #             kk = input_label[k]+1
    #             gt.append(kk.data.item()-1)
    #             #ipdb.set_trace()
    #             dist = []
    #             for jj in range(len(D)):
    #                 z_k = z_real[k,:].numpy().astype('float64').T.reshape(-1,1)
    #                 coeff_k = coeff_fun(z_k, D[jj])
    #                 z_recon = np.dot(D[jj], coeff_k)
    #                 error = np.linalg.norm(z_k - z_recon)
    #                 dist.append(error)
    #             min_dist = np.array(dist).min()
    #             dist_list.append(min_dist)
    #             all_count += 1
    #             #print('processing ')  
               
    #             if kk.item() in unseen_labels.tolist():
    #                 unseen_all +=1
    #                 if min_dist>threshold: 
    #                     out = self.zsl_classifier(input_k.view(1,-1))
    #                     out = convertTensor(out)
    #                     pred_label_ = torch.argmax(out,1)
    #                     pred_label = self.data.unseen_labels[pred_label_.data.item()]-1
    #                     pred.append(pred_label)
    #                     unseen_count +=1
    #                 else:
    #                     pred.append(1000)
                    
                    
    #             elif kk.item() in seen_labels.tolist():
    #                 seen_all +=1            
    #                 if min_dist<=threshold:
    #                     seen_count +=1
    #                     out = self.classifier(z_real[k,:].view(1,-1))
    #                     out = convertTensor(out)
    #                     pred_label = torch.argmax(out,1).data.item()
    #                     #pred_label = self.data.test_seen_labels[pred_label_.cpu().data.item()]-1
    #                     pred.append(pred_label)
    #                 else:
    #                     pred.append(1000) 
    #     pred_ = np.vstack(pred)
    #     gt_ = np.vstack(gt)
    #     acc = self.compute_acc(gt_, pred_ )

    #     mean_dist = mean_dist /all_count
    #     #ipdb.set_trace()
    #     return unseen_count/unseen_all, seen_count/seen_all , acc, dist_list

    def testing_2(self, epoch, test_class = 'seen', dataset = None, threshold = 0.99):
        if test_class == 'seen':
            test_loader = self.test_loader_seen
        elif test_class == 'unseen':
            test_loader = self.test_loader_unseen
        
        all_attrs = torch.Tensor(dataset.attrs).float().cuda()
        seen_labels = dataset.seen_labels
        unseen_labels = dataset.unseen_labels
        
        if isinstance(threshold, np.ndarray):
            thresholds = threshold
        else:
            thresholds = np.ones(seen_labels.shape[0]) * threshold
        
        self.load_models(epoch)        
        z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
        all_anchors = self.attr_encoder(all_attrs)[0]
        all_anchors = convertTensor(all_anchors)  
        seen_idx = seen_labels - 1
        unseen_idx = unseen_labels -1
        
        seen_anchors = all_anchors[seen_idx.tolist(),:]
        unseen_anchors = all_anchors[unseen_idx.tolist(),:]
        
        seen_count = 0
        seen_all = 1
        unseen_count = 0
        unseen_all = 1
        all_count = 0
        min_thres = 10
        mean_dist = 0
        dist_list = []
        pred = []
        gt = []
        for i_batch, sample_batched in enumerate(test_loader):
            input_data = sample_batched['feature']
            input_label = sample_batched['label']   
            input_attr = sample_batched['attr']

            ######
            input_data = convertTensor(input_data)
            input_label = convertTensor(input_label)
            input_attr = convertTensor(input_attr)

            batch_size = input_data.size()[0]

            input_data = input_data.astype('float32')
            input_label = input_label.astype('int64') 
            input_attr = input_attr.astype('float32')
                                
            m, s = self.encoder(input_data)   
            m = convertTensor(m)
            s = convertTensor(s)
            # z_real = self.reparameterize(m, s).rsample().squeeze()
            z_real = m.squeeze()
            
            for k in range(z_real.shape[0]):
                input_k = input_data[k,:]
                kk = input_label[k]+1
                gt.append(kk.data.item()-1)
                z_tile = z_real[k,:].repeat(seen_anchors.shape[0]).view(seen_anchors.shape[0],-1)
                dist = F.cosine_similarity(z_tile, seen_anchors)
                max_idx = torch.argmax(dist)
                ######
                max_idx = tuple(max_idx)

                mean_dist += dist.max()
                dist_list.append(dist.max().item())
                all_count += 1  
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        unseen_count +=1
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1  
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1
                
                '''
                if kk.item() in unseen_labels.tolist():
                    unseen_all +=1
                    if dist.max()<thresholds[max_idx]: 
                        out = self.zsl_classifier(input_k.view(1,-1))
                        out = convertTensor(out)
                        # out = self.zsl_classifier.test(input_k.view(1,-1))
                        pred_label_ = torch.argmax(out,1)
                        ######
                        # pred_label = self.data.unseen_labels[pred_label_.cpu().data.item()]-1
                        pred_label = self.data.unseen_labels[pred_label_.data.item()]-1
                        # out = self.classifier(z_real[k,:].view(1,-1))
                        # pred_label = torch.argmax(out,1).data.item()
                        pred.append(pred_label)
                        unseen_count +=1
                    else:
                        pred.append(1000)
                                       
                elif kk.item() in seen_labels.tolist():
                    seen_all +=1            
                    if dist.max()>=thresholds[max_idx]:
                        seen_count +=1
                        out = self.classifier(z_real[k,:].view(1,-1))
                        out = convertTensor(out)
                        pred_label = torch.argmax(out,1).data.item()
                        #pred_label = self.data.test_seen_labels[pred_label_.cpu().data.item()]-1
                        pred.append(pred_label)
                    else:
                        pred.append(1000)
                                                    
        pred_ = np.vstack(pred)
        gt_ = np.vstack(gt)
        acc = self.compute_acc(gt_, pred_ )

        mean_dist = mean_dist /all_count
        return unseen_count/unseen_all, seen_count/seen_all , acc, dist_list
        
    def draw_roc_curve(self, epoch, data):
        import sklearn.metrics as metrics
        threshold = self.search_thres_by_traindata(epoch, dataset = data, n = 0.99)
        unseen_acc, _, ts, dist_unseen = self.testing_2(epoch, test_class ='unseen', dataset = data, threshold = threshold)
        _, seen_acc,  tr, dist_seen = self.testing_2(epoch, test_class ='seen', dataset = data, threshold = threshold)
         
        print('fpr = {}, tpr = {}'.format(100*(1-unseen_acc), 100*seen_acc)) 
        print("ts {}  tr {}, H {}".format(ts, tr, 2*ts*tr/(ts + tr)))
        # ipdb.set_trace()
        dists = np.concatenate((np.array(dist_unseen), np.array(dist_seen)))
        
        labels_unseen = np.zeros(len(dist_unseen))
        labels_seen = np.ones(len(dist_seen))
        
        labels = np.concatenate((labels_unseen, labels_seen))
        fpr, tpr, threshold = metrics.roc_curve(labels, dists)
        roc_auc = metrics.auc(fpr, tpr)
        
        print('fpr = {}, tpr = {}, auc = {}'.format(0, 0, 100*roc_auc))
        
        with open("{}_res.pkl".format(self.dataset_name), 'wb') as f:      
            pickle.dump({'fpr': fpr, 'tpr':tpr}, f)  
            f.close()
            print('save data done!')
            
        plt.title('ROC curves on the 5 benchmark datasets')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        # ipdb.set_trace()
        exit()
        return 0   
    
    # def testing(self, epoch, if_viz = True, sample_rate = 2):
    #     file_encoder = 'Checkpoint_{}_Enc.pdparams.tar'.format(epoch)
    #     file_decoder = 'Checkpoint_{}_Dec.pdparams.tar'.format(epoch)
    #     file_attr_encoder = 'Checkpoint_{}_attr_Enc.pdparams.tar'.format(epoch)
        
    #     enc_path = os.path.join(self.save_path, file_encoder)
    #     dec_path = os.path.join(self.save_path, file_decoder)
    #     attr_enc_path = os.path.join(self.save_path, file_attr_encoder)
    
    #     enc_checkpoint = torch.load(enc_path)
    #     self.encoder.set_state_dict(enc_checkpoint['state_dict'])
        
    #     dec_checkpoint = torch.load(dec_path)
    #     self.decoder.set_state_dict(dec_checkpoint['state_dict'])
        
    #     attr_enc_checkpoint = torch.load(attr_enc_path)
    #     self.attr_encoder.set_state_dict(attr_enc_checkpoint['state_dict'])
         
    #     self.encoder.eval()
    #     self.decoder.eval()
    #     self.attr_encoder.eval()
             
    #     z = []; label = []; recon = []; data_in = []; z_attr = []; muu = []; sigmaa = []
    #     '''
    #     class_names = ["antelope", "grizzly bear", "killer whale", "beaver", "dalmatian", "persian cat", "horse",
    #                        "german shepherd", "blue whale", "siamese cat", "skunk",  "mole", "tiger", "hippopotamus",
    #                        "leopard", "moose", "spider monkey", "humpback whale", "elephant", "gorilla", "ox",  "fox",
    #                        "sheep", "seal" ,"chimpanzee", "hamster", "squirrel", "rhinoceros", "rabbit", "bat", "giraffe",
    #                        "wolf", "chihuahua", "rat", "weasel","otter", "buffalo", "zebra", "giant panda", "deer", "bobcat",
    #                        "pig", "lion", "mouse", "polar bear", "collie", "walrus", "raccoon", "cow", "dolphin"]
    #     '''
    #     class_names = ["Seen Features","Unseen Features"]                   
        
    #     for i_batch, sample_batched in enumerate(self.test_loader_unseen):
    #         input_data = sample_batched['feature']
    #         input_label = sample_batched['label']   
    #         input_attr = sample_batched['attr']

    #         ######
    #         input_data = convertTensor(input_data)
    #         input_label = convertTensor(input_label)
    #         input_attr = convertTensor(input_attr)

    #         batch_size = input_data.size()[0]
            
    #         input_data = input_data.astype('float32')
    #         input_label = input_label.astype('int64')  
    #         input_attr = input_attr.astype('float32')
            
    #         if self.ifsample:
    #             m, s = self.encoder(input_data)
    #             z_real = self.reparametrize(m, s)
    #         else:
    #             z_real = self.encoder(input_data)[0]
    #         z_real = convertTensor(z_real)
            
    #         x_recon = self.decoder(z_real)
    #         x_recon = convertTensor(x_recon)
                
    #         mu, sigma = self.attr_encoder(input_attr)
    #         z_fake = self.reparameterize(mu, sigma).rsample().squeeze()
    #         z_fake = convertTensor(z_fake)
       
    #         muu.append(z_fake.squeeze().numpy())
            
    #         z.append(z_real.numpy())
    #         label.append(input_label.numpy().reshape(-1,1))
    #         recon.append(x_recon.numpy())
    #         data_in.append(input_data.numpy())
    #         z_attr.append(z_fake.squeeze().numpy())
            
    #         recon_loss = self.criterion(x_recon, input_data)
    #         recon_loss = torch.dot(z_real[1,:], z_fake[1,:])
    #         print('batch {} recon_loss = {}'.format(i_batch, recon_loss))
        
    #     muu_ = np.vstack(muu)      
    #     z_ = np.vstack(z)
    #     recon_ = np.vstack(recon)
    #     label_ = np.vstack(label).reshape(-1)
    #     data_in_ = np.vstack(data_in)
    #     z_attr_ = np.vstack(z_attr)
      
    #     if if_viz:
    #         from sklearn.manifold import TSNE
    #         from matplotlib import colors as mcolors

    #         colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    #         color_list = []
    #         for color in colors.keys():
    #             if color == 'aliceblue':
    #                 color_list.append('y')
    #             elif color == 'k':
    #                 color_list.append('purple')
    #             else:
    #                 color_list.append(color)
            
    #         color_list[0] = 'blue'
    #         color_list[1] = 'darkorange'
            
    #         label_colors = []
    #         label_names = []
     
    #         for i in range(len(z_attr_)):
    #             label_colors.append(color_list[label_[i]])
    #             label_names.append(class_names[label_[i]])
          
        
    #         model = TSNE(n_components = 2, n_iter = 5000, init = 'pca',random_state = 0)
               
    #         #zz_ = np.vstack([z_, muu_])
    #         zz_ = np.vstack([z_, z_])
    #         label_colors__ = label_colors
    #         label_colors = label_colors + label_colors__      
    #         z_sample = zz_[range(0,zz_.shape[0],sample_rate),:]
    #         label_colors_sample = label_colors[::sample_rate] 
    #         label_names_sample = label_names[::sample_rate] 

    #         z_2d = model.fit_transform(z_sample)
    #         fig = plt.figure(figsize = (12, 12) )
    #         ax = fig.add_subplot(111)
    #         n = z_2d.shape[0]
            
    #         df1 = pd.DataFrame({"x": z_2d[0:n//2, 0], "y": z_2d[0:n//2, 1], "colors": label_colors_sample[0:n//2]})
    #         for i, dff in df1.groupby("colors"):
    #             class_name = class_names[color_list.index(i)]
    #             plt.scatter(dff['x'], dff['y'], c=i, label= class_name, marker = '.')
              
    #         ax.scatter(z_2d[0:n//2, 0], z_2d[0:n//2, 1], c=label_colors_sample[0:n//2] , marker = '.')
    #         ax.scatter(z_2d[n//2:n, 0], z_2d[n//2:n, 1], c=label_colors_sample[n//2:n], marker = '.')
    #         ax.set_facecolor('gray')
    #         #ax.set_ylim(-48, 48)
    #         #ax.set_xlim(-48, 48)
    #         plt.axis('off')
    #         #ax.set_yticklabels([])
    #         #ax.set_xticklabels([])
            
    #         box = ax.get_position()
    #         ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
    #         ax.legend(fontsize = 'xx-large',loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    #         #plt.legend(fontsize = "small", loc=1)
    #         plt.show()
    #         plt.savefig('awa2.jpg')
    #         ipdb.set_trace()
    #     return z_, recon_, label       
