# encdoing: utf-8
"""
@Project: DevCoder20201214
@File:    ARE
@Author:  Jiachen Zhao
@Time:    2020/12/29 15:02
@Description: The ARE model for anomaly detection,
can use fully-connected layers and CNN layers, *****DOWN*****
can back to the deep autoencoder,
could be trained under semi-supervised and unsupervised,
can use weighted sampler or not *****DOWN*****
can weight the reconstruction loss or not
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models.algorithm_utils import PyTorchUtils
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.neighbors import NearestNeighbors
from SyntheticData.pyodData import SYNTHETICDATASET
from sklearn.manifold import TSNE
import time


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 16, 4, 4)


def buildEncoderDecoder(nntype='fc', **kwargs):
    """
    :param nntype: str 'fc' for fully-connected encoder or 'cnn' for convolution nn, CNN autoencoder is for Mnist or FashionMnist dataset
    :return: torch.sequential model, encoder and decoder
    """
    if nntype == 'fc':
        layer_dims = kwargs['layer_dims']
        if type(layer_dims) is list:
            layer_dims = np.array(layer_dims)
        # encoder
        encoder_dims = np.concatenate([[layer_dims[0]], layer_dims[1:-1].repeat(2), [layer_dims[-1]]])
        encoder_layers = np.array(
            [[nn.Linear(int(a), int(b)), nn.ReLU()] for a, b in encoder_dims.reshape(-1, 2)]).flatten()[:-1]
        # print(encoder_layers)
        # print(encoder_layers)
        encoder = nn.Sequential(*encoder_layers)
        # print(encoder)
        # decoder
        decoder_dims = encoder_dims[::-1]
        decoder_layers = np.array(
            [[nn.Linear(int(a), int(b)), nn.ReLU()] for a, b in decoder_dims.reshape(-1, 2)]).flatten()[:-1]
        decoder = nn.Sequential(*decoder_layers)
        return encoder, decoder
    elif nntype == 'cnn':
        hidden_dims = kwargs['hidden_dims']
        assert hidden_dims >= 2, 'Hidden dimension should be lareger than 2'
        conv1 = nn.Conv2d(1, 6, 5, stride=2)
        conv2 = nn.Conv2d(6, 16, 5, stride=2)
        fc120 = nn.Linear(16 * 4 * 4, 120)
        encoder_layers = [conv1, conv2, Flatten(), fc120, nn.ReLU()]

        t_conv1 = nn.ConvTranspose2d(in_channels=6, out_channels=1,
                                     kernel_size=6, stride=2)
        t_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=6,
                                     kernel_size=6, stride=2)
        t_fc120 = nn.Linear(120, 16 * 4 * 4)

        decoder_layers = [nn.ReLU(), t_fc120, UnFlatten(), t_conv2, t_conv1]

        if hidden_dims <= 50:
            encoder_layers.append(nn.Linear(120, 50))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Linear(50, hidden_dims))
            decoder_layers.insert(0, nn.Linear(50, 120))
            decoder_layers.insert(0, nn.ReLU())
            decoder_layers.insert(0, nn.Linear(hidden_dims, 50))
        elif hidden_dims > 50:
            encoder_layers.append(nn.Linear(120, hidden_dims))
            decoder_layers.insert(0, nn.Linear(hidden_dims, 120))
        # if hidden_dims <= 10:
        #     encoder_layers.append(nn.Linear(50, 10))
        #     encoder_layers.append(nn.ReLU())
        #     decoder_layers.insert(0, nn.Linear(10, 50))
        # elif 50 > hidden_dims > 10:
        #     encoder_layers.append(nn.Linear(50, hidden_dims))
        #     decoder_layers.insert(0, nn.Linear(hidden_dims, 50))
        # if hidden_dims <= 2:
        #     encoder_layers.append(nn.Linear(10, 2))
        #     decoder_layers.insert(0, nn.Linear(2, 10))
        # elif 10 > hidden_dims > 2:
        #     encoder_layers.append(nn.Linear(10, hidden_dims))
        #     decoder_layers.insert(0, nn.Linear(hidden_dims, 10))
        encoder = nn.Sequential(*encoder_layers)
        decoder = nn.Sequential(*decoder_layers)
        return encoder, decoder


def computeSamplingWeight(y):
    """
    Compute the sampling weight for the dataloader to balance the number of anomaly and normal data
    :param y: numpy.ndarray or troch.tensor, labels
    :return: w_sampling: torch.tensor with shape like y
    """
    anomaly_percentage = 1 / 2
    if type(y) is np.ndarray:
        y = torch.from_numpy(y).float()
    elif torch.is_tensor(y):
        pass
    w_sampling = torch.ones_like(y, dtype=float)
    outlier_index = torch.nonzero(y, as_tuple=False)[:, 0]
    inlier_index = torch.nonzero((y == 0), as_tuple=False)[:, 0]
    w_sampling[outlier_index] = anomaly_percentage / outlier_index.shape[0]
    w_sampling[inlier_index] = (1. - anomaly_percentage) / inlier_index.shape[0]
    w_sampling = w_sampling.view([-1])
    return w_sampling

def computeReconWeight(y):
    if type(y) is np.ndarray:
        y = torch.from_numpy(y).float()
    elif torch.is_tensor(y):
        pass
    w_reconloss = torch.ones_like(y, dtype=float) - y
    return w_reconloss



def compute_peak_ind(Z, n_neighbors=3):
    '''
    Compute the local reachability density for Z
    :param Z: [num_samples, dimensions]
    :return:
    '''
    if torch.is_tensor(Z):
        Z = Z.cpu().detach().numpy()
    neigh = NearestNeighbors(n_neighbors=n_neighbors, p=2)
    neigh.fit(Z)
    '''计算最近邻图'''
    neigh_graph = neigh.kneighbors_graph(Z, n_neighbors=n_neighbors, mode='distance')
    neigh_graph.eliminate_zeros()
    '''计算最近邻的距离和标签'''
    neigh_dist, neigh_ind = neigh.kneighbors(Z)
    # neigh_dist [n_sample, n_neighbors], each row is the distance of the sample to its neighbors
    # neigh_ind [n_sample, n_neighbors], each row is the index of the sample's neighbors

    '''计算可达距离， reachability distance'''
    dist_k = neigh_dist[neigh_ind, n_neighbors - 1]
    reach_dist_array = np.maximum(neigh_dist, dist_k)
    # reach_dist_array[i,j] 是第i个样本的第j个neighbor到样本i的可达距离
    '''计算局部可达密度'''
    lrd = 1. / (np.mean(reach_dist_array, axis=1) + 1e-10)
    # local_reachability_density是可达距离平均值的倒数
    # print(neigh_ind)
    # print(np.sum(np.eye(len(lrd))[neigh_ind], axis=1))
    # peak_ind = np.where(np.max(np.sum(np.eye(len(lrd))[neigh_ind], axis=1) * lrd, axis=1) == lrd)[0]
    threshold = np.percentile(lrd, 50)
    peak_ind = np.array([i for i in range(len(lrd)) if ((lrd[i] > threshold) and (lrd[i] >= lrd[neigh_ind[i]]).all())])
    # print(f'num_peaks: {len(peak_ind)}')
    return peak_ind


class AREmodel(nn.Module, PyTorchUtils):
    def __init__(self, name='ARE', nnType=None, supervisedMode='semi',
                 n_neighbors=100, lamb=1., rho=5., margin=1.,
                 num_epochs=20, batch_size=64, lr=1e-3, pretrainFlag=True, seed=1, gpu=1, **kwargs):
        """
        :param name:
        :param nnType: str, 'fc' or 'cnn'
        :param n_neighbors:
        :param lamb: float, the trade-off parameter for loss function
        :param rho:
        :param margin:
        :param num_epochs:
        :param batch_size:
        :param lr:
        :param pretrainFlag: boolean, whether pretrain the model
        :param seed:
        :param gpu:
        :param kwargs: if nnType is 'fc',  then layer_dims: list;
                                    'cnn', then hidden_dims:int

                       landDistMode: str 'min' or 'appro', landmark distance mode
                        'loss': 'recon'             unsupervised autoencoder
                                'recon+attracting'  unsupervised reconstruction and attracting normal samples
                                'recon+ar'          haven't weighted sampling, directly reconstruct all samples and ar loss
                                'wt-recon+ar'       weighted sampling and rewighted recon loss


        """
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.name = f"{supervisedMode}"+name+f"_{kwargs['landDistMode']}_{lamb}_{rho}_{margin}"
        self.nnType = nnType
        self.n_neighbors = n_neighbors
        self.lamb = lamb  #
        self.rho = rho  # the smooth parameter used to approximate the minimum function
        self.margin = margin  # the distance margin for repelling loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.pretrainFlag = pretrainFlag
        self.supervisedMode = supervisedMode
        self.kwargs = kwargs
        # self.encoder, self.decoder = self.buildNet()
        # self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        # self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)

    def printInfo(self):
        if self.supervisedMode == 'un':
            modeName = 'Unsupervised'
        elif self.supervisedMode == 'FakeUn:':
            modeName = 'FakeUnsupervised'
        elif self.supervisedMode == 'semi':
            modeName = 'Semi-supervised'

        print('*' * 100)
        print(f"{self.name}, {modeName}, landDistMode={self.kwargs['landDistMode']}, loss={self.kwargs['loss']}")
        print('Kay parameters:')
        print(f'n_neighbors: {self.n_neighbors}')
        print(f'lamb: {self.lamb} \t The trade-off parameter for loss function')
        print(f'rho: {self.rho}   \t The smooth parameter used to approximate the minimum function')
        print(f'margin: {self.margin} \t The distance margin for repelling loss')
        print('*' * 100)

    def buildNet(self):
        if self.nnType == 'fc':
            encoder, decoder = buildEncoderDecoder(nntype='fc', layer_dims=self.kwargs['layer_dims'])
        elif self.nnType == 'cnn':
            encoder, decoder = buildEncoderDecoder(nntype='cnn', hidden_dims=self.kwargs['hidden_dims'])
        self.to_device(encoder)
        self.to_device(decoder)
        return encoder, decoder

    def to_tensor(self, X=None, y=None):
        if X is not None:
            if type(X) is np.ndarray:
                X = self.to_var(torch.from_numpy(X).float())
            elif torch.is_tensor(X):
                X = self.to_var(X.float())
        if y is not None:
            if type(y) is np.ndarray:
                y = self.to_var(torch.from_numpy(y).int())
            elif torch.is_tensor(y):
                y = self.to_var(y.int())
        return X, y

    def initLandmarks(self, TSNE_embedding=True, *args):
        X = args[0]

        if len(X.shape) > 2:
            X_temp = X.reshape(X.shape[0], -1)
        else:
            X_temp = X

        if TSNE_embedding:
            peakInd = compute_peak_ind(TSNE(n_components=2,random_state=np.random.RandomState(1)).fit_transform(X_temp), n_neighbors=self.n_neighbors)
        else:
            peakInd = compute_peak_ind(X_temp, n_neighbors=self.n_neighbors)
        print(f'peakInd: {peakInd}')
        if len(args) == 2:
            y = args[1]
            if any(y[peakInd] == 1):
                print(f'!!!There exist anomaly being initilized as landmarks!!!')
                peakInd = peakInd[y[peakInd] == 0]
        landmarks = X[peakInd]
        if len(landmarks.shape) == 3:
            landmarks = landmarks.unsqueeze(0)
        landmarks_embedded = self.embedding(landmarks)
        print(f'landmarks_embedded shape:{landmarks_embedded.shape}')
        return landmarks_embedded, landmarks, peakInd

    def landmarkSelect(self, *args):
        # assert self.pretrained_flag == 1, "Model is not pretrained, so can't select landmarks"
        X = args[0]
        assert X.shape[0] > self.n_neighbors, 'n_neighbor is too large!'
        Z = self.embedding(X)
        peakInd = compute_peak_ind(Z.cpu().detach().numpy(), n_neighbors=self.n_neighbors)
        if len(args) > 1:
            y = args[1]
            if torch.is_tensor(y):
                y = y.cpu().detach().numpy()
            try:
                if any(y[peakInd] == 1):
                    print(f'!!!There exist anomaly being selected as landmarks!!!')
                    peakInd = peakInd[y[peakInd] == 0]
            except:
                pass
        landmarks = X[peakInd]
        print(f'selceted peakInd{peakInd}')
        landmarks_embedded = Z[peakInd]
        return landmarks_embedded, landmarks, peakInd

    def computeDist2Landmarks(self, enc, landmarks, mode='min'):
        if landmarks is None:
            landmarks = torch.zeros(enc.shape[1])
        else:
            assert enc.shape[1] == landmarks.shape[1], "enc and landmarks dimension doesn't match"
        dist_landmarks = torch.cdist(enc, landmarks, 2)
        if mode == 'min':
            return dist_landmarks.min(dim=1).values
        elif mode == 'appro':
            return (-1 / self.rho * torch.log(torch.sum(torch.exp(-self.rho * dist_landmarks),
                                                        dim=1))).abs()  # to approximate the min function of a vector

    def embedding(self, X):
        X, _ = self.to_tensor(X)
        self.encoder.eval()
        enc = self.encoder.forward(X)
        # enc = self.normalizeLayer.forward(enc)
        return enc

    def loss_function(self, *args):
        """
        :param args: [x, dec, enc, landmarks, y, w_reconloss]
        :return:
        """
        if len(args) == 2:
            # Unsupervised autoencoder with only reconstruction loss, loss code: "U1A00"
            x, dec = args[0], args[1]
            loss_recon = torch.mean(torch.pow(x - dec, 2))
            return loss_recon
        if len(args) == 4:
            # Unsupervised attracting loss
            x, dec, enc, landmarks = args[0], args[1], args[2], args[3]
            loss_recon = torch.mean(torch.pow(x - dec, 2))
            dist_landmarks = self.computeDist2Landmarks(enc, landmarks, mode=self.kwargs['landDistMode'])
            loss_normal = dist_landmarks.mean()
            return loss_recon, loss_normal
        if len(args) == 5:
            # semi-supervised attracting-repelling loss for unweighted sampling with unweighted recon loss

            x, dec, enc, landmarks, y = args[0], args[1], args[2], args[3], args[4]
            loss_recon = torch.mean(torch.mean(torch.pow(x - dec, 2), dim=1))
            dist_landmarks = self.computeDist2Landmarks(enc, landmarks, mode=self.kwargs['landDistMode'])
            loss_normal = ((1. - y) * dist_landmarks).mean()
            loss_abnormal = (y.float() * F.relu(self.margin - dist_landmarks)).mean()
            return loss_recon, loss_normal, loss_abnormal

        if len(args) == 6:
            # Semi-supervised attracting-repelling loss with weighted reconstruction loss
            x, dec, enc, landmarks, y, w_reconloss = args[0], args[1], args[2], args[3], args[4], args[5]
            loss_recon = torch.mean(w_reconloss * torch.pow(x - dec, 2).view(x.size(0), -1).mean(1))
            dist_landmarks = self.computeDist2Landmarks(enc, landmarks, mode=self.kwargs['landDistMode'])
            loss_normal = ((1. - y) * dist_landmarks).mean()
            loss_abnormal = (y.float() * F.relu(self.margin - dist_landmarks)).mean()
            return loss_recon, loss_normal, loss_abnormal

    def fit(self, *args):
        """
        :param X: np.array with shape of [num_sample, feature_dimensions, ...], training samples
        :param y: np.array with shape of [num_sample, ], training labels
        :return:
        """
        self.encoder, self.decoder = self.buildNet()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)
        if self.pretrainFlag:
            self.pretrain(args[0])
        if self.supervisedMode == 'un':
            X = args[0]
            landmarks = self.unsupervisedFit(X)
        elif self.supervisedMode == 'FakeUn':
            X = args[0]
            landmarks = self.unsupervisedFit(X)
        elif self.supervisedMode == 'semi':
            landmarks = self.semisupervisedFit(args[0], args[1])
        return landmarks

    def pretrain(self, X):
        X, _ = self.to_tensor(X, None)
        train_dataloader = DataLoader(dataset=TensorDataset(X),
                                      batch_size=self.batch_size, drop_last=False)
        preoptimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
        for epoch in range(10):
            self.encoder.train()
            self.decoder.train()
            pre_losses = []
            for [X_batch] in train_dataloader:
                X_batch = self.to_var(X_batch)
                preoptimizer.zero_grad()
                enc = self.encoder.forward(X_batch)
                dec = self.decoder.forward(enc)
                loss_pre = torch.mean(torch.pow(X_batch - dec, 2))
                loss_pre.backward()
                preoptimizer.step()
                pre_losses.append(loss_pre.cpu().detach().numpy())
            if epoch % 5 == 0 or epoch == 9:
                print(f'pretrain epoch {epoch}: {np.mean(pre_losses):0.4f}')

    def unsupervisedFit(self, X):
        if self.kwargs['loss'] == 'recon+attracting':
            landmarks_embedded, landmarks, peakInd = self.initLandmarks(True, X)
        X, _ = self.to_tensor(X, None)
        train_dataloader = DataLoader(dataset=TensorDataset(X),
                                      batch_size=self.batch_size, drop_last=False)
        for epoch in range(self.num_epochs):
            self.encoder.train()
            self.decoder.train()
            recon_losses, normal_losses, total_losses = [], [], []

            for [X_batch] in train_dataloader:
                X_batch = self.to_var(X_batch)
                self.optimizer.zero_grad()
                enc = self.encoder.forward(X_batch)
                dec = self.decoder.forward(enc)
                if self.kwargs['loss'] == 'recon':
                    loss_recon = self.loss_function(X_batch, dec)
                    loss_normal = 0
                    normal_losses.append(loss_normal)
                elif self.kwargs['loss'] == 'recon+attracting':
                    loss_recon, loss_normal = self.loss_function(X_batch, dec, enc, self.to_var(landmarks_embedded))
                    normal_losses.append(loss_normal.cpu().detach().numpy())
                loss = loss_recon + self.lamb * loss_normal
                loss.backward()
                self.optimizer.step()

                recon_losses.append(loss_recon.cpu().detach().numpy())
                total_losses.append(loss.cpu().detach().numpy())
            if self.kwargs['loss'] == 'recon+attracting':
                if epoch % 5 == 0:
                    landmarks_embedded, _, _ = self.landmarkSelect(X)
            self.scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f'epoch: {epoch + 1}\t'
                      f'loss_recon: {np.mean(recon_losses):0.4f}\t'
                      f'loss_normal: {np.mean(normal_losses):0.4f}\t'
                      f'loss: {np.mean(total_losses):0.4f}')

        return self.landmarkSelect(X)[0] if self.kwargs['loss'] == 'recon+attracting' else None

    def semisupervisedFit(self, X, y):
        landmarks_embedded, landmarks, peakInd = self.initLandmarks(True, X, y)
        X, y = self.to_tensor(X, y)
        if self.kwargs['loss'] == 'wt-recon+ar':
            w_sampling = computeSamplingWeight(y)
            # w_reconloss = 1. - w_sampling
            w_reconloss = computeReconWeight(y)
            sampler = WeightedRandomSampler(weights=w_sampling,
                                            num_samples=len(y) * 2,  # Sampler size is twice of dataset,
                                            # because set anomaly_percentage = 1/2 in computeSamplingWeight,
                                            replacement=True)
            train_dataloader = DataLoader(dataset=TensorDataset(X, y, w_reconloss),
                                          batch_size=self.batch_size,
                                          sampler=sampler, drop_last=False)
        elif self.kwargs['loss'] == 'recon+ar':
            train_dataloader = DataLoader(dataset=TensorDataset(X, y),
                                          batch_size=self.batch_size, drop_last=False)
        for epoch in range(self.num_epochs):
            self.encoder.train()
            self.decoder.train()
            recon_losses, normal_losses, abnormal_losses, ar_losses, total_losses = [], [], [], [], []
            for Batch in train_dataloader:
                X_btach, y_batch = self.to_var(Batch[0]), self.to_var(Batch[1])
                if len(Batch) == 3:
                    w_reconloss_batch = self.to_var(Batch[2])
                self.optimizer.zero_grad()
                enc = self.encoder.forward(X_btach)
                dec = self.decoder.forward(enc)
                if self.kwargs['loss'] == 'recon+ar':
                    loss_recon, loss_normal, loss_abnormal = self.loss_function(X_btach, dec, enc,
                                                                                self.to_var(landmarks_embedded),
                                                                                y_batch)
                elif self.kwargs['loss'] == 'wt-recon+ar':
                    loss_recon, loss_normal, loss_abnormal = self.loss_function(X_btach, dec, enc,
                                                                                self.to_var(landmarks_embedded),
                                                                                y_batch, w_reconloss_batch)
                loss_ar = loss_normal + loss_abnormal
                loss = loss_recon + self.lamb * loss_ar
                loss.backward()
                self.optimizer.step()
                recon_losses.append(loss_recon.cpu().detach().numpy())
                normal_losses.append(loss_normal.cpu().detach().numpy())
                abnormal_losses.append(loss_abnormal.cpu().detach().numpy())
                ar_losses.append(loss_ar.cpu().detach().numpy())
                total_losses.append(loss.cpu().detach().numpy())
            if epoch % 5 == 0:
                landmarks_embedded, _, _ = self.landmarkSelect(X, y)
            self.scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f'epoch: {epoch + 1}\t'
                      f'loss_recon: {np.mean(recon_losses):0.4f}\t'
                      f'loss_normal: {np.mean(normal_losses):0.4f}\t'
                      f'loss_abnormal: {np.mean(abnormal_losses):0.4f}\t'
                      f'loss_ar: {np.mean(ar_losses):0.4f}\t'
                      f'loss: {np.mean(total_losses):0.4f}')
        landmarks_embedded, _, _ = self.landmarkSelect(X, y)
        return landmarks_embedded

    def predict(self, X, landmarks=None):
        X, _ = self.to_tensor(X)
        self.encoder.eval()
        self.decoder.eval()
        enc = self.encoder.forward(X)
        dec = self.decoder.forward(enc)
        score_recon = torch.pow(X - dec, 2).view(X.size(0), -1).mean(1).cpu().detach().numpy()
        # score_recon = torch.mean(torch.pow(X - dec, 2), dim=1).cpu().detach().numpy()
        if landmarks is not None:
            # score_dist = self.computeDist2Landmarks(enc, landmarks, mode='min').cpu().detach().numpy()
            score_distappro = self.computeDist2Landmarks(enc, landmarks, mode='appro').cpu().detach().numpy()
            return score_recon, score_distappro, score_recon + self.lamb * score_distappro
        else:
            return None, None, score_recon


if __name__ == "__main__":
    print('---')
    # y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    # w_sampling = computeSamplingWeight(y)
    # w_reconloss = 1 - w_sampling
    # print(w_sampling)
    # print(w_sampling.sum())
    # print(w_reconloss)
    # print(w_reconloss.sum())
    # X = torch.randn([10, 3])
    # y = torch.tensor(y)
    #
    # train_dataloader = DataLoader(dataset=TensorDataset(X, y, w_reconloss),
    #                               batch_size=5, drop_last=False)
    # for [x_batch, y_batch, w_reconloss_batch] in train_dataloader:
    #     print(x_batch)
    #     print(y_batch)
    #     print(w_reconloss_batch)
    # model = AREmodel(nnType='fc', layer_dims=[5, 2], landDistMode='min')
    # Z = np.random.random([10000, 2])
    # start = time.time()
    #
    # n_neighbors=200
    # neigh = NearestNeighbors(n_neighbors=n_neighbors, p=2)
    # neigh.fit(Z)
    # '''计算最近邻图'''
    # neigh_graph = neigh.kneighbors_graph(Z, n_neighbors=n_neighbors, mode='distance')
    # neigh_graph.eliminate_zeros()
    # '''计算最近邻的距离和标签'''
    # neigh_dist, neigh_ind = neigh.kneighbors(Z)
    # # neigh_dist [n_sample, n_neighbors], each row is the distance of the sample to its neighbors
    # # neigh_ind [n_sample, n_neighbors], each row is the index of the sample's neighbors
    #
    # '''计算可达距离， reachability distance'''
    # dist_k = neigh_dist[neigh_ind, n_neighbors - 1]
    # reach_dist_array = np.maximum(neigh_dist, dist_k)
    # # reach_dist_array[i,j] 是第i个样本的第j个neighbor到样本i的可达距离
    # '''计算局部可达密度'''
    # lrd = 1. / (np.mean(reach_dist_array, axis=1) + 1e-10)
    # # local_reachability_density是可达距离平均值的倒数
    # peakInd = [i for i in range(len(lrd)) if (lrd[i] >= lrd[neigh_ind[i]]).all()]
    # print(peakInd)
    # print(neigh_ind[0].shape)
    # print(lrd[neigh_ind[0]])
    # print((lrd[0] >= lrd[neigh_ind[0]]).all())
    # # print(np.sum(np.eye(len(lrd))[neigh_ind], axis=1))
    # print(lrd.shape)
    # end = time.time()
    # print(end-start)
    x = np.random.random([100000000, 2])
    start = time.time()
    # compute_peak_ind(x, n_neighbors=300)
    a = True or (x > 0.000001).any()
    end = time.time()
    print(end - start)
    x = np.random.random([100000000, 2])
    start = time.time()
    # compute_peak_ind(x, n_neighbors=300)
    a = (x > 0.000001).any()
    end = time.time()
    print(end - start)
