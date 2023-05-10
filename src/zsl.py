USE_CLASS_STANDARTIZATION = True # i.e. equation (9) from the paper
USE_PROPER_INIT = True # i.e. equation (10) from the paper

from time import time

import numpy as np
import paddle
import paddle_torch as torch
import paddle_torch.nn as nn
import paddle_torch.nn.functional as F
from paddle_torch.tensor import convertTensor
from paddle_torch.utils.data import DataLoader
from tqdm import tqdm


class Pre_train():
    def __init__(self,dataraw,dataset):
        np.random.seed(1)
        torch.manual_seed(1)

        #####
        class ClassStandardization(nn.Module):
            """
            Class Standardization procedure from the paper.
            Conceptually, it is equivalent to nn.BatchNorm1d with affine=False,
            but for some reason nn.BatchNorm1d performs slightly worse.
            """
            def __init__(self, feat_dim: int):
                super().__init__()
                self.running_mean = torch.zeros(feat_dim)
                self.running_var = torch.ones(feat_dim)

            def forward(self, class_feats):
                """
                Input: class_feats of shape [num_classes, feat_dim]
                Output: class_feats (standardized) of shape [num_classes, feat_dim]
                """
                if self.training:
                    batch_mean = class_feats.mean(dim=0)
                    batch_var = class_feats.var(dim=0)
                        
                    # Normalizing the batch
                    result = (class_feats - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-5)
                        
                    # Updating the running mean/std
                    self.running_mean = 0.9 * self.running_mean.detach() + 0.1 * batch_mean.detach()
                    self.running_var = 0.9 * self.running_var.detach() + 0.1 * batch_var.detach()

                else:
                    # Using accumulated statistics
                    # Attention! For the test inference, we cant use batch-wise statistics,
                    # only the accumulated ones. Otherwise, it will be quite transductive
                    result = (class_feats - self.running_mean.unsqueeze(0)) / (self.running_var.unsqueeze(0) + 1e-5)
                return convertTensor(result)


        class CNZSLModel(nn.Module):
            def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int):
                super().__init__()
                self.attention = nn.Linear(proto_dim,proto_dim)
                self.model = nn.Sequential(
                    nn.Linear(attr_dim, hid_dim),
                    nn.ReLU(),
                        
                    nn.Linear(hid_dim, hid_dim),
                    ClassStandardization(hid_dim),
                    nn.ReLU(),
                        
                    ClassStandardization(hid_dim),
                    nn.Linear(hid_dim, proto_dim),
                    nn.ReLU(),
                    )
                    
                # if USE_PROPER_INIT:
                weight_var = 1 / (hid_dim * proto_dim)
                b = np.sqrt(3 * weight_var)
                self.model[-2].weight = paddle.create_parameter(shape=[hid_dim, proto_dim], dtype='float32',
                                                                attr=paddle.ParamAttr(name=None, initializer=paddle.nn.initializer.Uniform(low=-b, high=b)),
                                                                is_bias=False)
                
                    
            def forward(self, x, attrs):
                atten = F.softmax(self.attention(x),dim=1)
                # atten = self.attention(x)
                x = x+x*atten
                protos = self.model(attrs)
                # protos = atten*protos
                # 5
                x_ns = 4 * x / x.norm(dim=1, keepdim=True) # [batch_size, x_dim]
                protos_ns = 4 * protos / protos.norm(dim=1, keepdim=True) # [num_classes, x_dim]
                logits = x_ns @ protos_ns.t() # [batch_size, num_classes]
                
                return logits

        ###
        DATASET = dataset
        DEVICE = 'cuda'
        data = dataraw.rawdata
        attrs_mat = dataraw.attrs_mat

        feats = data['features'].T.astype(np.float32)
        # for i in range(feats.shape[0]):
        #     feats[i,:] = feats[i,:]/np.linalg.norm(feats[i,:]) * 1.0
        labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
        train_idx = attrs_mat['trainval_loc'].squeeze() - 1
        test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
        test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1
        test_idx = np.array(test_seen_idx.tolist() + test_unseen_idx.tolist())
        seen_classes = sorted(np.unique(labels[test_seen_idx]))
        unseen_classes = sorted(np.unique(labels[test_unseen_idx]))

        print(f'<=============== Preprocessing ===============>')
        num_classes = len(seen_classes) + len(unseen_classes)
        seen_mask = np.array([(c in seen_classes) for c in range(num_classes)])
        unseen_mask = np.array([(c in unseen_classes) for c in range(num_classes)])
        attrs = attrs_mat['att'].T
        attrs = torch.from_numpy(attrs).float()
        attrs = attrs / attrs.norm(dim=1, keepdim=True) * np.sqrt(attrs.shape[1])
        attrs_seen = attrs[seen_mask]
        attrs_unseen = attrs[unseen_mask]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        test_seen_idx = [i for i, y in enumerate(test_labels) if y in seen_classes]
        test_unseen_idx = [i for i, y in enumerate(test_labels) if y in unseen_classes]
        labels_remapped_to_seen = [(seen_classes.index(t) if t in seen_classes else -1) for t in labels]
        test_labels_remapped_seen = [(seen_classes.index(t) if t in seen_classes else -1) for t in test_labels]
        test_labels_remapped_unseen = [(unseen_classes.index(t) if t in unseen_classes else -1) for t in test_labels]
        ds_train = [(feats[i], labels_remapped_to_seen[i]) for i in train_idx]
        ds_test = [(feats[i], int(labels[i])) for i in test_idx]
        train_dataloader = DataLoader(ds_train, batch_size=256, shuffle=True)
        test_dataloader = DataLoader(ds_test, batch_size=2048)
        cls_ds_test = [(feats[i], labels_remapped_to_seen[i]) for i in test_idx if labels_remapped_to_seen[i] >=0]
        cls_test_dataloader = DataLoader(cls_ds_test, batch_size=2048)

        dds_train = [(feats[i], int(labels[i])) for i in train_idx]
        dtrain_dataloader = DataLoader(dds_train, batch_size=256, shuffle=True)
        class_indices_inside_test = {c: [i for i in range(len(test_idx)) if labels[test_idx[i]] == c] for c in range(num_classes)}


        print(f'\n<=============== Starting training ===============>')
        start_time = time()
        model = CNZSLModel(attrs.shape[1], 1024, feats.shape[1])

        scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.0005, step_size=25, gamma=0.1)
        optim = torch.optim.Adam(lr=scheduler, params=model.parameters(), weight_decay=0.0001)
        # criterion = paddle.nn.CrossEntropyLoss()
        # seen_cls = c2.CLASSIFIER(copy.deepcopy(dtrain_dataloader),(num_classes),test_dataloader,test_labels,class_indices_inside_test,seen_classes,unseen_classes)
        # optim = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)


        for epoch in tqdm(range(50)):
            model.train()
            
            for i, batch in enumerate(train_dataloader):
                feats = torch.from_numpy(np.array(batch[0]))
                targets = torch.from_numpy(np.array(batch[1])).astype('int64')
                logits = model(feats, attrs[seen_mask])

                loss = F.cross_entropy(logits, targets)

                optim.zero_grad()
                loss.backward()
                optim.step()
            
            scheduler.step()

        print(f'Training is done! Took time: {(time() - start_time): .1f} seconds')

        model.eval() # Important! Otherwise we would use unseen batch statistics
        logits = [model(convertTensor(x), convertTensor(attrs)) for x, _ in test_dataloader]
        logits = torch.cat(logits, dim=0)
        # logits[:, seen_mask] *= (0 if DATASET != "CUB" else 0) # Trading a bit of gzsl-s for a bit of gzsl-u
        preds_gzsl = logits.argmax(dim=1).detach().numpy()
        preds_zsl_s = logits[:, seen_mask].argmax(dim=1).detach().numpy()
        preds_zsl_u = logits[:, ~seen_mask].argmax(dim=1).detach().numpy()
        guessed_zsl_u = (preds_zsl_u == test_labels_remapped_unseen)
        guessed_gzsl = (preds_gzsl == test_labels)
        zsl_unseen_acc = np.mean([guessed_zsl_u[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]]) 
        gzsl_seen_acc = np.mean([guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in seen_classes]])
        gzsl_unseen_acc = np.mean([guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]])
        gzsl_harmonic = 2 * (gzsl_seen_acc * gzsl_unseen_acc) / (gzsl_seen_acc + gzsl_unseen_acc)

        print(f'ZSL-U: {zsl_unseen_acc * 100:.02f}')
        print(f'GZSL-U: {gzsl_unseen_acc * 100:.02f}')
        print(f'GZSL-S: {gzsl_seen_acc * 100:.02f}')
        print(f'GZSL-H: {gzsl_harmonic * 100:.02f}')

        self.model = model
        self.attrs = attrs
        self.seen_mask = seen_mask
        self.unseen_mask = unseen_mask
    

    def test(self,input):
        logits = self.model(input,self.attrs)
        logits = convertTensor(logits)
        logits[:,self.seen_mask] = 0
        # return logits
        return logits[:, self.unseen_mask]
