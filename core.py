import os

import torch
import torch.optim as optim
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score

import params
from utils import make_variable


def train_src(encoder, classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()
    lossi = []

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):

        for step, (samples, labels) in enumerate(data_loader):

            # check if GPU is present
            if torch.cuda.is_available():
                samples = samples.cuda()
                labels = labels.cuda()

            # zero gradients for optimizer
            optimizer.zero_grad()
            # compute loss for encoder
            loss = criterion(classifier(encoder(samples)), labels)
            lossi.append(loss.item())
            # optimize source classifier
            loss.backward()
            optimizer.step()

        # print epoch info
        if ((epoch + 1) % params.log_step == 0):
            print("Epoch [{}/{}] Step [{}/{}]: loss={:.5f}"
                    .format(epoch + 1,
                            params.num_epochs_pre,
                            step + 1,
                            len(data_loader),
                            loss.item()))


    fig = plt.figure()  # figsize=(6, 6)
    ax = fig.add_subplot(111)
    plt.plot(torch.tensor(lossi).view(-1, params.log_step).mean(1))
    plt.title('Training source loss')
    plt.savefig('loss.png', bbox_inches='tight', dpi=600)

    return encoder, classifier


def eval_src(encoder, classifier, data_loader, fig_title=None):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    feat = np.array([]).reshape(0,2)
    label_pred = np.array([], dtype=int)
    label_true = np.array([], dtype=int)

    # evaluate network
    for (samples, labels) in data_loader:

        # make smaples and labels variable
        samples = make_variable(samples)
        labels = make_variable(labels)

        preds = classifier(encoder(samples))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

        feat = np.vstack((feat,samples.detach().numpy()))
        label_pred = np.hstack((label_pred,preds.data.max(1)[1].detach().numpy()))
        label_true = np.hstack((label_true,labels.numpy()))

    loss /= len(data_loader)
    acc = acc.item()/len(data_loader.dataset)
    ari = adjusted_rand_score(label_true.flatten(), label_pred.flatten())

    print("Avg Loss = {}, Avg Accuracy = {:2%}, ARI = {:.5f}".format(loss, acc, ari))

    if fig_title is not None:
        h = .02  # step size in the mesh
        x_min, x_max = samples[:, 0].min() - 5, samples[:, 0].max() + 5
        y_min, y_max = samples[:, 1].min() - 5, samples[:, 1].max() + 5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        xxyy = np.concatenate((xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1)),axis=1)
        xxyy = torch.from_numpy(xxyy).to(torch.float32)
        Z = classifier(encoder(xxyy))
        Z = Z.max(1)[1].detach().numpy()
        Z = Z.reshape(xx.shape)        


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(fig_title)

        ax1.scatter(feat[:, 0], feat[:, 1], c=label_true, s=40, cmap=plt.cm.coolwarm)
        ax1.set_xlim(xx.min(), xx.max())
        ax1.set_ylim(yy.min(), yy.max())

        ax2.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
        ax2.scatter(feat[:, 0], feat[:, 1], c=label_pred, s=40, cmap=plt.cm.coolwarm)
        ax2.set_xlim(xx.min(), xx.max())
        ax2.set_ylim(yy.min(), yy.max())

        ax1.set_title('true labels')
        ax2.set_title('predicted labels')
        ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)    
        plt.savefig(fig_title+'.png', bbox_inches='tight', dpi=600)


def train_tgt(src_encoder, tgt_encoder, discriminator, classifier, src_data_loader, tgt_data_loader):
    """
    Adversarial adaptation to train target encoder.
    Train encoder for target domain.
    """
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers    
    discriminator.train()
    tgt_encoder.train()    
    generator_lossi = []
    discriminator_lossi = []

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_discriminator = optim.Adam(discriminator.parameters(),
                                         lr=params.d_lr,
                                         betas=(params.beta1, params.beta2))
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.g_lr,
                               betas=(params.beta1, params.beta2))
    # A scheduler to reduce the learning rate
    scheduler_tgt = optim.lr_scheduler.MultiStepLR(optimizer_tgt,
                                                   milestones=[round(params.num_epochs*0.3),round(params.num_epochs*0.6)], gamma=1e-1)
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
            
        # zip source and target data pair
        for step, ((samples_src, _), (samples_tgt, _)) in enumerate(zip(src_data_loader, tgt_data_loader)):

            # make images variable
            if torch.cuda.is_available():
                samples_src = samples_src.cuda()
                samples_tgt = samples_tgt.cuda()

            # extract and concat features
            feat_src = src_encoder(samples_src)
            feat_tgt = tgt_encoder(samples_tgt)
            # detach feat_tgt from the tgt_encoder to avoid this error:
            # RuntimeError: Trying to backward through the graph a second time
            feat_concat = torch.cat((feat_src, feat_tgt.detach()), 0)

            # prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0), requires_grad=False).long())
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0), requires_grad=False).long())
            adversary_label = torch.cat((label_src, label_tgt), 0)

            ############################
            # 2.2 train target encoder #
            ############################

            # clear out the gradients from the last step loss
            optimizer_tgt.zero_grad()
            # compute loss for target encoder
            generator_loss = criterion(discriminator(feat_tgt), 1 - label_tgt)
            generator_lossi.append(generator_loss.item())
            # backward propagation: calculate gradients
            generator_loss.backward()
            # update the weights
            optimizer_tgt.step()

            ###########################
            # 2.1 train discriminator #
            ###########################

            # clear out the gradients from the last step loss.
            optimizer_discriminator.zero_grad()            
            # compute loss for discriminator
            discriminator_loss = criterion(discriminator(feat_concat), adversary_label)
            discriminator_lossi.append(discriminator_loss.item())
            #backward propagation: calculate gradients
            discriminator_loss.backward()
            #update the weights
            optimizer_discriminator.step()


            

        #######################
        # 2.3 print epoch info #
        #######################
        if ((epoch + 1) % params.log_step == 0):
            print("Epoch [{}/{}] Step [{}/{}]:"
                    "d_loss={:.5f} g_loss={:.5f}"
                    .format(epoch + 1,
                            params.num_epochs,
                            step + 1,
                            len_data_loader,
                            discriminator_loss.item(),
                            generator_loss.item()))
        
        # step both schedulers
        # scheduler_discriminator.step()
        scheduler_tgt.step()

    fig = plt.figure()  # figsize=(6, 6)
    ax = fig.add_subplot(111)
    plt.plot(torch.tensor(discriminator_lossi).view(-1, params.log_step).mean(1))
    plt.title('Discriminator loss')
    plt.savefig('discriminator_loss.png', bbox_inches='tight', dpi=600)

    fig = plt.figure()  # figsize=(6, 6)
    ax = fig.add_subplot(111)
    plt.plot(torch.tensor(generator_lossi).view(-1, params.log_step).mean(1))
    plt.title('Generator loss')
    plt.savefig('generator_loss.png', bbox_inches='tight', dpi=600)

    return tgt_encoder
