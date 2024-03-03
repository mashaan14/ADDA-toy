import os
import math
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
    loss_e = []
    loss_min = float('inf')

    # setup criterion and optimizer
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                           lr=s_lr)
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(num_epochs_pre):
      loss_i = torch.zeros(len(XS_train_dataloader))
      for step, (samples, labels) in enumerate(data_loader):
        # Transfer to GPU
        samples, labels = samples.to(device), labels.to(device)
        # zero gradients for optimizer
        optimizer.zero_grad()
        # compute loss for encoder
        loss = criterion(classifier(encoder(samples)), labels)
        loss_i[step] = loss
        # optimize source classifier
        loss.backward()
        optimizer.step()

      if loss_i.mean() < loss_min:
        loss_min = loss_i.mean()
        encoder_loss_min = encoder
        classifier_loss_min = classifier
      loss_e.append(loss_i.mean().item())
      # print epoch info
      if ((epoch + 1) % log_step == 0):
          print("Epoch [{}/{}]: loss={:.4f}"
                  .format(epoch + 1,
                          num_epochs_pre,
                          loss_i.mean().item()))


    fig = plt.figure()  # figsize=(6, 6)
    ax = fig.add_subplot(111)
    # plot the average loss over 100 epochs
    # plt.plot(torch.tensor(lossi).view(-1, 10).mean(1))
    plt.plot(loss_e)
    plt.title('Training source loss')
    plt.savefig('loss.png', bbox_inches='tight', dpi=600)

    return encoder_loss_min, classifier_loss_min


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
        h = .01  # step size in the mesh
        x_min, x_max = feat[:, 0].min() - 5, feat[:, 0].max() + 5
        y_min, y_max = feat[:, 1].min() - 5, feat[:, 1].max() + 5
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

    d_loss_e = []
    g_loss_e = []
    g_loss_min = float('inf')

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=d_lr)
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(), lr=g_lr)

    ####################
    # 2. train network #
    ####################

    for epoch in range(num_epochs):
      d_loss_i = torch.zeros(len(XT_train_dataloader))
      g_loss_i = torch.zeros(len(XT_train_dataloader))
      # zip source and target data pair
      for step, ((samples_src, _), (samples_tgt, _)) in enumerate(zip(src_data_loader, tgt_data_loader)):
        # Transfer to GPU
        samples_src, samples_tgt = samples_src.to(device), samples_tgt.to(device)

        # extract and concat features
        feat_src = src_encoder(samples_src)
        feat_tgt = tgt_encoder(samples_tgt)
        # detach feat_tgt from the tgt_encoder to avoid this error:
        # RuntimeError: Trying to backward through the graph a second time
        feat_concat = torch.cat((feat_src, feat_tgt.detach()), 0)

        # prepare real and fake label
        label_src = torch.ones(feat_src.size(0), requires_grad=False).long().to(device)
        label_tgt = torch.zeros(feat_tgt.size(0), requires_grad=False).long().to(device)
        adversary_label = torch.cat((label_src, label_tgt), 0)

        ###########################
        # 2.1 train discriminator #
        ###########################

        # clear out the gradients from the last step loss.
        optimizer_discriminator.zero_grad()
        # compute loss for discriminator
        discriminator_loss = criterion(discriminator(feat_concat), adversary_label)
        d_loss_i[step] = discriminator_loss
        #backward propagation: calculate gradients
        discriminator_loss.backward()
        #update the weights
        optimizer_discriminator.step()

        ############################
        # 2.2 train target encoder #
        ############################

        # clear out the gradients from the last step loss
        optimizer_tgt.zero_grad()
        # compute loss for target encoder
        generator_loss = criterion(discriminator(feat_tgt), 1 - label_tgt)
        # generator_loss = criterion(discriminator(feat_concat.detach()), 1 - adversary_label)
        g_loss_i[step] = generator_loss
        # backward propagation: calculate gradients
        generator_loss.backward()
        # update the weights
        optimizer_tgt.step()

      d_loss_e.append(d_loss_i.mean().item())
      if g_loss_i.mean() < g_loss_min:
        g_loss_min = g_loss_i.mean()
        tgt_encoder_loss_min = tgt_encoder
      g_loss_e.append(g_loss_i.mean().item())
      #######################
      # 2.3 print epoch info #
      #######################
      if ((epoch + 1) % log_step == 0):
          print("Epoch [{}/{}]: d_loss={:.4f} g_loss={:.4f}"
                  .format(epoch + 1,
                          num_epochs,
                          d_loss_i.mean().item(),
                          g_loss_i.mean().item()))

    fig = plt.figure()  # figsize=(6, 6)
    ax = fig.add_subplot(111)
    # plot the average loss
    # plt.plot(torch.tensor(discriminator_lossi).view(-1, log_step).mean(1))
    plt.plot(d_loss_e)
    plt.title('Discriminator loss')
    plt.savefig('discriminator_loss.png', bbox_inches='tight', dpi=600)

    fig = plt.figure()  # figsize=(6, 6)
    ax = fig.add_subplot(111)
    # plot the average loss
    # plt.plot(torch.tensor(generator_lossi).view(-1, log_step).mean(1))
    plt.plot(g_loss_e)
    plt.title('Generator loss')
    plt.savefig('generator_loss.png', bbox_inches='tight', dpi=600)

    return tgt_encoder_loss_min
