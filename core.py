import os

import torch
import torch.optim as optim
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

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

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):

        if epoch+1 == params.num_epochs_pre:
            encoded_feat = np.zeros((len(data_loader), 2))
            label_pred = np.zeros((len(data_loader), 1), dtype=int)
            label_true = np.zeros((len(data_loader), 1), dtype=int)

        for step, (samples, labels) in enumerate(data_loader):

            if epoch+1 == params.num_epochs_pre:
                encoded_feat[step, :] = encoder(samples).detach().numpy()
                label_pred[step, :] = classifier(encoder(samples)).data.max(1)[1].detach().numpy()
                label_true[step, :] = labels.numpy()

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for encoder
            preds = classifier(encoder(samples))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.item()))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src(encoder, classifier, data_loader)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Encoding features for source data')
    for g in np.unique(label_true):
        ix = np.where(label_true == g)
        ax1.scatter(encoded_feat[ix, 0], encoded_feat[ix, 1])

    for g in np.unique(label_pred):
        ix = np.where(label_pred == g)
        ax2.scatter(encoded_feat[ix, 0], encoded_feat[ix, 1])

    ax1.set_title('true labels')
    ax2.set_title('predicted labels')
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    plt.savefig('plot-XS-train.png', bbox_inches='tight', dpi=600)

    return encoder, classifier


def eval_src(encoder, classifier, data_loader, fig_title):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    feat = np.zeros((len(data_loader), 2))
    label_pred = np.zeros((len(data_loader), 1), dtype=int)
    label_true = np.zeros((len(data_loader), 1), dtype=int)
    step = 0

    # evaluate network
    for (samples, labels) in data_loader:

        preds = classifier(encoder(samples))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

        feat[step, :] = samples.detach().numpy()
        label_pred[step, :] = preds.data.max(1)[1].detach().numpy()
        label_true[step, :] = labels.numpy()
        step += 1

    loss /= len(data_loader)
    acc = acc.item()/len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(fig_title)
    for g in np.unique(label_true):
        ix = np.where(label_true == g)
        ax1.scatter(feat[ix, 0], feat[ix, 1])

    for g in np.unique(label_pred):
        ix = np.where(label_pred == g)
        ax2.scatter(feat[ix, 0], feat[ix, 1])

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
    tgt_encoder.train()
    discriminator.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_discriminator = optim.Adam(discriminator.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):

        if epoch + 1 <= params.num_epochs:
            encoded_feat = np.zeros((len(tgt_data_loader), 2))
            label_pred = np.zeros((len(tgt_data_loader), 1), dtype=int)
            label_true = np.zeros((len(tgt_data_loader), 1), dtype=int)

        # zip source and target data pair
        for step, ((samples_src, _), (samples_tgt, labels_tgt)) in enumerate(zip(src_data_loader, tgt_data_loader)):
            ###########################
            # 2.1 train discriminator #
            ###########################

            # zero gradients for optimizer
            optimizer_discriminator.zero_grad()

            # extract and concat features
            feat_src = src_encoder(samples_src)
            feat_tgt = tgt_encoder(samples_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = discriminator(feat_concat.detach())

            # prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)
            # label_concat = torch.stack((label_src, label_tgt), 0)

            # compute loss for discriminator
            loss_discriminator = criterion(pred_concat, label_concat)
            loss_discriminator.backward()

            # optimize discriminator
            optimizer_discriminator.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            if epoch+1 <= params.num_epochs:
                encoded_feat[step, :] = tgt_encoder(samples_tgt).detach().numpy()
                label_pred[step, :] = classifier(tgt_encoder(samples_tgt)).data.max(1)[1].detach().numpy()
                label_true[step, :] = labels_tgt.numpy()

            # zero gradients for optimizer
            optimizer_discriminator.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(samples_tgt)

            # predict on discriminator
            pred_tgt = discriminator(feat_tgt)

            # prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len_data_loader,
                              loss_discriminator.item(),
                              loss_tgt.item(),
                              acc.item()))

        # plot visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Encoding features for target data')
        for g in np.unique(label_true):
            ix = np.where(label_true == g)
            ax1.scatter(encoded_feat[ix, 0], encoded_feat[ix, 1])

        for g in np.unique(label_pred):
            ix = np.where(label_pred == g)
            ax2.scatter(encoded_feat[ix, 0], encoded_feat[ix, 1])

        ax1.set_title('true labels')
        ax2.set_title('predicted labels')
        ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.savefig('plot-XT-train'+str(epoch)+'.png', bbox_inches='tight', dpi=600)

    return tgt_encoder
