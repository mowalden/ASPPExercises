"""A class intended to check whether a cymetric model obeys certain symmetries.
"""

import os as os
import pickle as pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.models.callbacks import RicciCallback, SigmaCallback, VolkCallback, KaehlerCallback, TransitionCallback
from cymetric.models.tfmodels import MultFSModel, PhiFSModel
from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, VolkLoss, RicciLoss, TotalLoss


class SymmetryCheck():
    """Checks specified symmetries of a specified hypersurface.

    The class takes a trained model and evaluates the model at points before and after a certain symmetry transformation. 
    It then computes the difference between those points and averages over all points.

    Attributes:
        symmetry: Symmetry to be checked. Available are permutation and roots_of_unity.
        model: Trained model. If none, the class trains a model itself, based on the hypersurface definition.
        hypersurface: List given by [monomical, coefficients, kmoduli, ambient] that define the hypersurface.
                      Can be left empty, only if trained model and points are given.
        points: Points of the hypersurface. If none, the model generates points based on the hypersurface definition.
    """
    
    def __init__(self, symmetry, model=None, hypersurface = None, points=None):
        """Initializes the instances based on the symmetry to be checked, the model, hypersurface and points.
        
        Args:
            symmetry: Symmetry to be checked.
            model: Trained model.
            hypersurface: List of hypersurface definitions.
            points: Points of hypersurface.
        """

        self.dirname = 'data'
        self.hypersurface = hypersurface

        if symmetry != 'permutation' and symmetry != 'roots_of_unity':
            raise ValueError(f'Symmetry {symmetry} not allowed. Must be either \'permutation\' or \'roots_of_unity\'.')
        else:
            self.symmetry = symmetry
        if model == None:
            if hypersurface == None:
                raise Exception('Model and hypersurface cannot be both undefined.')
            else:
                cb_list, BASIS = self.prepare_data()
                self.model, self.training_history = self.train_model(cb_list, BASIS)

        else:
            self.model = model
            self.training_history = None
                

    def prepare_data(self):
        """Creates data needed for the model to be trained based on the hypersurface input.

        Returns:
            cb: Callbacks needed for training the model.
            BASIS: Basis needed for training the model.
        """

        try:
            self.pg = PointGenerator(self.hypersurface[0], self.hypersurface[1], self.hypersurface[2], self.hypersurface[3])
        except:
            raise ValueError(f'Invalid input for hypersurface: {self.hypersurface}')
        try:
            kappa = self.pg.prepare_dataset(10000, self.dirname)
            self.pg.prepare_basis(self.dirname, kappa=kappa)
        except:
            raise Exception('Error during creation of data folder. Please make sure that there is no folder with the name \'data\' in your directory.')
        self.data = np.load(os.path.join(self.dirname, 'dataset.npz'))
        BASIS = np.load(os.path.join(self.dirname, 'basis.pickle'), allow_pickle=True)
        BASIS = prepare_tf_basis(BASIS)

        # Define callbacks
        ricci_cb = RicciCallback((self.data['X_val'], self.data['y_val']), self.data['val_pullbacks'])
        sigma_cb = SigmaCallback((self.data['X_val'], self.data['y_val']))
        volk_cb = VolkCallback((self.data['X_val'], self.data['y_val']))
        kaehler_cb = KaehlerCallback((self.data['X_val'], self.data['y_val']))
        transition_cb = TransitionCallback((self.data['X_val'], self.data['y_val']))
        cb_list = [ricci_cb, sigma_cb, volk_cb, kaehler_cb, transition_cb]

        return cb_list, BASIS

    def train_model(self, cb_list, BASIS):
        """Defines and trains model.

        Args:
            cb_list: List of callbacks needed for training the model.
            BASIS: Basis needed for training the model.

        Returns:
            fmodel: Training model.
            training_history = Training history of the model.
        """
        # Define neural network
        nlayer = 3
        nHidden = 64
        act = 'gelu'
        nEpochs = 50

        # The batch size, the second one is for VolkLoss Specifically
        bSizes = [64, 50000]

        # weights on the loss functions
        alpha = [1., 1., 1., 1., 1.]

        # Define input and output size
        n_in = 2*5
        n_out = 1

        # Setup layers
        nn = tf.keras.Sequential()
        nn.add(tf.keras.Input(shape=(n_in)))
        for i in range(nlayer):
            nn.add(tf.keras.layers.Dense(nHidden, activation=act))
        nn.add(tf.keras.layers.Dense(n_out, use_bias=False))

        # Define model
        fmodel = PhiFSModel(nn, BASIS, alpha=alpha)

        # Define loss functions
        cmetrics = [TotalLoss(), SigmaLoss(), KaehlerLoss(), TransitionLoss(), VolkLoss(), RicciLoss()]

        # Training
        opt = tf.keras.optimizers.Adam()
        fmodel, training_history = train_model(fmodel, self.data, optimizer=opt, epochs=nEpochs, batch_sizes=[64, 50000],
                                            verbose=1, custom_metrics=cmetrics, callbacks=cb_list)

        return fmodel, training_history
                

monomials = 5*np.eye(5, dtype=np.int64)
coefficients = np.ones(5)
kmoduli = np.ones(1)
ambient = np.array([4])

hypersurface = [monomials, coefficients, kmoduli, ambient]

test = SymmetryCheck('permutation', hypersurface=hypersurface)
# # Hypersurface definition, needed to check whether a point is on the CY
# monomials = 5*np.eye(5, dtype=np.int64)
# coefficients = np.ones(5)
# kmoduli = np.ones(1)
# ambient = np.array([4])
# pg = PointGenerator(monomials, coefficients, kmoduli, ambient)


# data = np.array([
#                     dict(np.load(os.path.join('fermat_pg', 'dataset.npz'))), 
#                     dict(np.load(os.path.join('fermat_pg_asym', 'dataset.npz')))
#                  ])

# model1 = tf.keras.models.load_model('fermat_pg/quintic.keras')
# model2 = tf.keras.models.load_model('fermat_pg_asym/quintic_asym.keras')
# BASIS1 = np.load(os.path.join('fermat_pg', 'basis.pickle'), allow_pickle=True)
# BASIS1 = prepare_tf_basis(BASIS1)
# BASIS2 = np.load(os.path.join('fermat_pg_asym', 'basis.pickle'), allow_pickle=True)
# BASIS2 = prepare_tf_basis(BASIS2)
# alpha = [1.,1.,1.,1.,1.]

# metric = np.array([ PhiFSModel(model1, BASIS1, alpha=alpha), 
#                     PhiFSModel(model2, BASIS2, alpha=alpha)
#                     ]) 

# examples = np.array([
#                         'Quintic with permutation symmetry', 
#                         'Another quinctic with permutation symmetry'
#                         ])

# # Number of examples
# n = np.size(examples, axis=0)
# # Patch
# patch = 0

# diff = np.zeros(3)
# diffGeneral = np.zeros(2)

# for i in range(n):
    
#     # Select patch
#     data[i]['X_train'] = np.delete(data[i]['X_train'], np.where(data[i]['X_train'][:,patch] != 1.), axis=0)

#     # Permutation together with selection of points that are all solved for the same coordinate
#     points = data[i]['X_train'][:, 0:pg.ncoords] + 1.j*data[i]['X_train'][:, pg.ncoords:]
#     solvedFor = metric[i]._find_max_dQ_coords(tf.convert_to_tensor(data[i]['X_train'], dtype=tf.float32))
#     idx = np.where(solvedFor == 1)
#     points = points[idx[0]]
#     pointsPerm = points.copy()
#     pointsPerm[:,2:5] = np.roll(pointsPerm[:,2:5], -1, axis=-1)

#     # Scale points
#     if i ==0:
#         scaleFactor = np.array([1., 1., np.exp(2*np.pi*1./5.j), np.exp(2*np.pi*2./5.j), 
#                                 np.exp(2*np.pi*3./5.j)])
#         scaleFactor = np.repeat(np.expand_dims(scaleFactor, axis=0), len(points), axis=0)
#         dataScale = points*scaleFactor
#         pointsScaled = np.concatenate((np.real(dataScale), np.imag(dataScale)), axis=-1)
#         pointsScaled = tf.convert_to_tensor(pointsScaled, dtype=tf.float32)

#     # Compute random difference
#     pointsFiltered = np.concatenate((np.real(points), np.imag(points)), axis=-1)
#     idx = np.random.randint(np.size(pointsFiltered, axis=0), size=400)
#     randomPoints = np.array(np.split(pointsFiltered[idx], 2, axis=0))
#     norm = tf.norm(metric[i](randomPoints[0]), axis=[-2,-1])
#     diffMatrix = tf.norm(metric[i](randomPoints[0])-metric[i](randomPoints[1]), axis=[-2,-1])
#     diffGeneral[i] = tf.math.reduce_mean(diffMatrix/norm).numpy()

#     # Compute Jacobian of permutation in patch zero: dz'/dz, where z' is the permuted coordinate
#     jacobianPerm = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]]).T
#     jacobianPerm = np.repeat(np.expand_dims(jacobianPerm, axis=0), np.size(pointsPerm, axis=0), axis=0)

#     ##########################################################################################################
#     # # Generate Jacobian for each point
#     # commonFactor = np.power(-1.-pointsPerm[np.arange(len(solvedFor)),np.remainder(solvedFor+1, 5)]**5-pointsPerm[np.arange(len(solvedFor)),np.remainder(solvedFor+2, 5)]**5-pointsPerm[np.arange(len(solvedFor)),np.remainder(solvedFor+3, 5)]**5, -4./5.)

#     # jacobian = np.repeat(np.expand_dims(np.array([[0.+0.j,0.+0.j,0.+0.j],[0.+0.j,0.+0.j,0.+0.j],[0.+0.j,0.+0.j,0.+0.j]]), axis=0), np.size(commonFactor, axis=0), axis=0)

#     # jacobian[np.arange(len(solvedFor)), np.remainder(3-solvedFor, 3)] = np.array([-pointsPerm[np.arange(len(solvedFor)),np.remainder(solvedFor+1, 5)]**4*commonFactor,-pointsPerm[np.arange(len(solvedFor)),np.remainder(solvedFor+2, 5)]**4*commonFactor,-pointsPerm[np.arange(len(solvedFor)),np.remainder(solvedFor+3, 5)]**4*commonFactor]).T

#     # jacobian[np.arange(len(solvedFor)),np.remainder(4-solvedFor, 3),1] = 1. + 0.j
#     # jacobian[np.arange(len(solvedFor)),np.remainder(5-solvedFor, 3),2] = 1. + 0.j
#     ###########################################################################################################


#     # Convert point back to cymetric format
#     pointsPerm = np.concatenate((np.real(pointsPerm), np.imag(pointsPerm)), axis=-1)
#     pointsPerm = tf.convert_to_tensor(pointsPerm, dtype=tf.float32)
#     points = np.concatenate((np.real(points), np.imag(points)), axis=-1)
#     points = tf.convert_to_tensor(points, dtype=tf.float32)



#     # Calculate transformed metric at each point
#     beforeTrans = metric[i](points)
#     afterPerm = jacobianPerm@metric[i](pointsPerm)@np.transpose(jacobianPerm, axes=(0,2,1))            

#     diff[2*i] = tf.math.reduce_mean(tf.norm(beforeTrans - afterPerm,  axis=[-2,-1])/tf.norm(beforeTrans)).numpy()
#     # Scaling
#     if i == 0:
#         # Compute jacobian
#         jacobianScale = np.array([[np.exp(2*np.pi*1./5.j), 0.+0.j, 0.+0.j], [0.+0.j, np.exp(2*np.pi*2./5.j), 0.+0.j], [0.+0.j, 0.+0.j, np.exp(2*np.pi*3./5.j)]])
#         jacobianScale = np.repeat(np.expand_dims(jacobianScale, axis=0), np.size(dataScale, axis=0), axis=0)

#         j_elim = tf.expand_dims(tf.ones(len(points), dtype=tf.int64), axis=-1)

#         afterScale = jacobianScale@metric[i](pointsScaled, j_elim=j_elim)@np.transpose(np.conjugate(jacobianScale), axes=(0,2,1))

#         diff[i+1] = tf.math.reduce_mean(tf.norm(beforeTrans - afterScale, axis=1)/tf.norm(beforeTrans, axis=1)).numpy()


# for i in range(n):
#     print('--------------------------------------------------------------')
#     print('\n' + examples[i] + ': \n')
#     print('Number of points: ' + str(np.size(data[i]['X_train'], axis=0)))
#     print('General difference: ' + str(diffGeneral[i]))
#     print('Error permutation: ' + str(diff[2*i]))
#     if i == 0:
#         print('Error scaling: ' + str(diff[i+1]) + '\n')
#     else:
#         print(' ')

# print('--------------------------------------------------------------')






