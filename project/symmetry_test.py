"""A class intended to check whether a cymetric model obeys certain symmetries.
"""

import os as os
import pandas as pd
import numpy as np
import tensorflow as tf
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.models.callbacks import (RicciCallback, SigmaCallback, VolkCallback,
                                       KaehlerCallback, TransitionCallback)
from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.metrics import (SigmaLoss, KaehlerLoss, TransitionLoss,
                                     VolkLoss, RicciLoss, TotalLoss)


class SymmetryCheck():
    """Checks specified symmetries of a specified hypersurface.

    The class takes a trained model and evaluates the model at points
    before and after a certain symmetry transformation. It then computes
    the difference between those points and averages over all points.

    Attributes:
        symmetry: Symmetry to be checked. Available are permutation and roots_of_unity.
        model: Trained model. If none, the class trains a model itself,
               based on the hypersurface definition.
        hypersurface: List given by [monomical, coefficients, kmoduli, ambient]
                      that define the hypersurface. Can be left empty, only if
                      trained model and points are given.
        points: Points of the hypersurface. If none, the model generates points
                based on the hypersurface definition.
    """

    def __init__(self, symmetry, model=None, hypersurface=None, points=None):
        """Initializes the instances based on the symmetry to be checked, the model,
           hypersurface and points.

        Args:
            symmetry: Symmetry to be checked.
            model: Trained model. If this is given, then the additional data must be
                   in a folder named 'trained_model'.
            hypersurface: List of hypersurface definitions.
            points: Points of hypersurface.
        """

        self.dirnames = ['data', 'trained_model']
        self.hypersurface = hypersurface
        # weights on the loss functions
        self.alpha = [1., 1., 1., 1., 1.]

        if symmetry not in {'permutation', 'roots_of_unity'}:
            raise ValueError(f'Symmetry {symmetry} not allowed. Must be either \'permutation\' '
                             'or \'roots_of_unity\'.')
        self.symmetry = symmetry
        if model is None:
            if self.hypersurface is None:
                raise Exception('Model and hypersurface cannot be both undefined.')
            cb_list, basis, data = self._prepare_data()
            self.model, self.training_history = self._train_model(cb_list, basis)
        else:
            data = dict(np.load(os.path.join(self.dirnames[1], 'dataset.npz')))
            basis = np.load(os.path.join(self.dirnames[1], 'basis.pickle'),
                            allow_pickle=True)
            basis = prepare_tf_basis(basis)
            self.model = PhiFSModel(model, basis, alpha = self.alpha)
            self.training_history = None

        # Select patch
        patch = 0
        self.points = np.delete(data['X_train'],
                                np.where(data['X_train'][:,patch] != 1.),
                                axis=0)
        self.n_points = self.points.shape[0]
        self.general_diff = self._check_general_diff(self.points)


    def _prepare_data(self):
        """Creates data needed for the model to be trained based on the hypersurface input.

        Returns:
            cb: Callbacks needed for training the model.
            BASIS: Basis needed for training the model.
        """

        try:
            self.pointgen = PointGenerator(self.hypersurface[0],
                                     self.hypersurface[1],
                                     self.hypersurface[2],
                                     self.hypersurface[3])
        except:
            raise ValueError(f'Invalid input for hypersurface: {self.hypersurface}')
        try:
            kappa = self.pointgen.prepare_dataset(10000, self.dirnames[0])
            self.pointgen.prepare_basis(self.dirnames[0], kappa=kappa)
        except:
            raise Exception('Error during creation of data folder. Please make sure that '
                            'there is no folder with the name \'data\' in your directory.')
        data = np.load(os.path.join(self.dirnames[0], 'dataset.npz'))
        basis = np.load(os.path.join(self.dirnames[0], 'basis.pickle'), allow_pickle=True)
        basis = prepare_tf_basis(basis)

        # Define callbacks
        ricci_cb = RicciCallback((self.data['X_val'],
                                  self.data['y_val']),
                                  self.data['val_pullbacks'])
        sigma_cb = SigmaCallback((self.data['X_val'], self.data['y_val']))
        volk_cb = VolkCallback((self.data['X_val'], self.data['y_val']))
        kaehler_cb = KaehlerCallback((self.data['X_val'], self.data['y_val']))
        transition_cb = TransitionCallback((self.data['X_val'], self.data['y_val']))
        cb_list = [ricci_cb, sigma_cb, volk_cb, kaehler_cb, transition_cb]
        return cb_list, basis, data

    def _train_model(self, cb_list, basis):
        """Defines and trains model.

        Args:
            cb_list: List of callbacks needed for training the model.
            BASIS: Basis needed for training the model.

        Returns:
            fmodel: Training model.
            training_history = Training history of the model.
        """
        # Define neural network
        network_properties = {
                    'n_layers': 3,  # Number of layers
                    'n_nodes': 64,  # Number of nodes
                    'n_epochs': 50, # Number of epochs
                    'n_in': 10,     # Input dimension
                    'n_out': 1,     # Output dimension
                    'act': 'gelu',   # Activation function,
                    'b_size': [64, 50000] # Batch size
        }
        # Setup layers
        neural_net = tf.keras.Sequential()
        neural_net.add(tf.keras.Input(shape=(network_properties['n_in'])))
        for i in range(network_properties['n_layers']):
            neural_net.add(tf.keras.layers.Dense(network_properties['n_nodes'], 
                                                 activation=network_properties['act']))
        neural_net.add(tf.keras.layers.Dense(network_properties['n_out'], use_bias=False))

        # Define model
        fmodel = PhiFSModel(neural_net, basis, alpha=self.alpha)

        # Define loss functions
        cmetrics = [TotalLoss(),
                    SigmaLoss(),
                    KaehlerLoss(),
                    TransitionLoss(),
                    VolkLoss(),
                    RicciLoss()]

        # Training
        opt = tf.keras.optimizers.Adam()
        fmodel, training_history = train_model(fmodel,
                                               self.data,
                                               optimizer=opt,
                                               epochs=network_properties['n_epochs'],
                                               batch_sizes=network_properties['b_size'],
                                               verbose=1,
                                               custom_metrics=cmetrics,
                                               callbacks=cb_list)
        return fmodel, training_history
    
    def _check_general_diff(self, points):
        """Checks the general differences among points on the hypersurfes by taking 
           the difference of randomly drawn points.

        Args:
            points: Points on which the model is evaluated.

        Returns:
            general_diff: Normalized general difference
        """
        idx = np.random.randint(np.size(points, axis=0), size=400)
        randomPoints = np.array(np.split(points[idx], 2, axis=0))
        norm = tf.norm(self.model(randomPoints[0]), axis=[-2,-1])
        unnormalized_diff = tf.norm(self.model(randomPoints[0])-self.model(randomPoints[1]), axis=[-2,-1])
        general_diff = tf.math.reduce_mean(unnormalized_diff/norm).numpy()
        return general_diff





monomials = 5*np.eye(5, dtype=np.int64)
coefficients = np.ones(5)
kmoduli = np.ones(1)
ambient = np.array([4])

hyper_surf = [monomials, coefficients, kmoduli, ambient]

test = SymmetryCheck('permutation', model=tf.keras.models.load_model('trained_model/quintic.keras'))

print(test.n_points)
print(test.general_diff)
