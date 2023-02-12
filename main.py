from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch

import models
import utils
import core


def rotate(xy, theta):
    """
    Returns a rotated set of points.
    """
    s = np.sin(theta * np.pi / 180)
    c = np.cos(theta * np.pi / 180)
    center_of_rotation = np.mean(xy, axis=0)

    xyr = np.zeros((xy.shape[0], xy.shape[1]))
    xyr[:, 0] = (c * (xy[:, 0]-center_of_rotation[0])) - (s * (xy[:, 1]-center_of_rotation[1])) + center_of_rotation[0]
    xyr[:, 1] = (s * (xy[:, 0]-center_of_rotation[0])) + (c * (xy[:, 1]-center_of_rotation[1])) + center_of_rotation[1]

    return xyr

XS = np.genfromtxt('data/dbmoon1000_Instances.csv', delimiter=",")
YS = np.genfromtxt('data/dbmoon1000_Labels.csv', delimiter=",")

XT = rotate(XS, 30)

XS_train, XS_test, YS_train, YS_test = train_test_split(XS, YS, test_size=0.2, random_state=42)
XT_train, XT_test, YT_train, YT_test = train_test_split(XT, YS, test_size=0.2, random_state=42)



fig = plt.figure()  # figsize=(6, 6)
ax = fig.add_subplot(111)
for g in np.unique(YS_train):
    ix = np.where(YS_train == g)
    ax.scatter(XS_train[ix, 0], XS_train[ix, 1], label='source class '+str(int(g)))

ax.scatter(XT[:, 0], XT[:, 1], c='k', label='target')

plt.legend(loc=0)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
plt.savefig('dataset.png', bbox_inches='tight', dpi=600)


XS_train_tensor = torch.Tensor(XS_train)
XS_test_tensor = torch.Tensor(XS_test)
YS_train_tensor = torch.LongTensor(YS_train)
YS_test_tensor = torch.LongTensor(YS_test)

XT_train_tensor = torch.Tensor(XT_train)
XT_test_tensor = torch.Tensor(XT_test)
YT_train_tensor = torch.LongTensor(YT_train)
YT_test_tensor = torch.LongTensor(YT_test)

XS_train_dataset = torch.utils.data.TensorDataset(XS_train_tensor, YS_train_tensor)
XS_test_dataset = torch.utils.data.TensorDataset(XS_test_tensor, YS_test_tensor)

XT_train_dataset = torch.utils.data.TensorDataset(XT_train_tensor, YT_train_tensor)
XT_test_dataset = torch.utils.data.TensorDataset(XT_test_tensor, YT_test_tensor)

XS_train_dataloader = torch.utils.data.DataLoader(XS_train_dataset)
XS_test_dataloader = torch.utils.data.DataLoader(XS_test_dataset)

XT_train_dataloader = torch.utils.data.DataLoader(XT_train_dataset)
XT_test_dataloader = torch.utils.data.DataLoader(XT_test_dataset)

src_encoder = models.Encoder()
src_encoder.apply(utils.init_weights)
src_classifier = models.Classifier()
src_classifier.apply(utils.init_weights)

src_encoder, src_classifier = core.train_src(src_encoder, src_classifier, XS_train_dataloader)
core.eval_src(src_encoder, src_classifier, XS_test_dataloader, fig_title='Testing source data using source encoder')


tgt_encoder = models.Encoder()
tgt_encoder.apply(utils.init_weights)

discriminator = models.Discriminator()
discriminator.apply(utils.init_weights)

tgt_encoder = core.train_tgt(src_encoder, tgt_encoder, discriminator, src_classifier, XS_train_dataloader, XT_train_dataloader)

""" eval target encoder on test set of target dataset """
print("=== Evaluating classifier for encoded target domain ===")
print(">>> Testing target data using source encoder <<<")
core.eval_src(src_encoder, src_classifier, XT_test_dataloader, fig_title='Testing target data using source encoder')
print(">>> Testing target data using target encoder <<<")
core.eval_src(tgt_encoder, src_classifier, XT_test_dataloader, fig_title='Testing target data using target encoder')

