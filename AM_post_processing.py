import numpy as np
from sklearn.decomposition import PCA
from sklearn import mixture
from qnl_analysis.analysis_utils import *
from labrad.units import us, ns, MHz, GHz
from base_classes import Measurement_Process
import copy
import warnings


class Meas_PCA_Rotation(Measurement_Process):
    """Performs PCA on each demod frequency in order to find the ideal rotation.
        Accepts:
            rotate_in_place - Boolean - defaults True - rotate the data without writing to post_proccessed_data
                            This is equivalent to rotating the phase of the IQ plane before taking the measurement
            clockwise - Boolean - defaults False -  decides the direction of the rotation
            data_to_pca - Str - defaults 'post_demod_data' - where the data is taken from in the measurement
                            typically this will either be post_processed_data or post_demod_data
            trigger_to_PCA - int - defaults 0 - which trigger in the measurement to take it from, typically
                            this is 1 for heralded data, or 0 for unheralded data
            pca_fit - sklearn PCA model object - this is so old fits can be loaded
            pca_rotation_angles - matrix from the PCA model - also so that old fits can be loaded
        
        the properties of this class can be seen and altered through the .properties dictionary.
        The 'pca_rotation_angles' in the properties dictionary is useful for setting the IQ rotations of the alazar
        So a typical use would be to do a rabi measurement, run the fit on it, use the rotations to fix your
        IQ rotations in your alazar measurement then you do not have to use this class again.
        """
    def __init__(self, rotate_in_place=True, demod_index=0, clockwise=False, data_to_pca='post_demod_data', trigger=0,
                 pca_fit={}, pca_rotation_angle={}):
        super(Meas_PCA_Rotation, self).__init__()
        self.properties['process_name'] = 'Meas_PCA_Rotation'
        self.properties['rotate_in_place'] = rotate_in_place
        self.properties['clockwise'] = clockwise
        self.properties['data_to_pca'] = data_to_pca
        self.properties['trigger'] = trigger
        self.properties['pca_fit'] = pca_fit
        self.properties['demod_index'] = demod_index
        self.properties['pca_rotation_angle'] = pca_rotation_angle

    def fit(self, measurement, **kwargs):
        self.properties.update(kwargs)

        data_to_use = measurement[self.properties['data_to_pca']].mean(2)  # Average over the repetitions
        data_to_use = data_to_use[self.properties['demod_index'], :, self.properties['trigger']]  # select the appropriate trigger

        pca = PCA(2, whiten=True)
        pca.fit(data_to_use.T)
        rotation = pca.components_

        rot_det = np.sign(np.linalg.det(rotation))
        rotation[0, 0] = rot_det * rotation[0, 0]
        rotation[0, 1] = rot_det * rotation[0, 1]  # mirror an eigenvector to get positive determinate 1

        if np.abs(rotation[0, 1]) > 0:
            rotation = np.sign(rotation[0, 1]) * rotation  # Flip sign to rotate counterclockwise

        # make sure the sign makes sense and we don't have an all negative matrix
        rotation = np.sign(rotation[0,0]) * rotation

        # flip as appropriate
        if self.properties['clockwise']:
            rotation = rotation.T

        # force our pca fit to our new rotation, cuz otherwise its just wrong
        pca.components_ = rotation

        # jam that stuff into the properties for record keeping
        self.properties['pca_fit'] = pca
        self.properties['pca_rotation_angle'] = np.angle(rotation[0, 0] + 1j * rotation[1, 0])

    def process(self, measurement, **kwargs):
        if 'Demod_Index_{}'.format(self.properties['demod_index']) not in measurement.keys():
            measurement['Demod_Index_{}'.format(self.properties['demod_index'])] = dict()

        if not self.properties['rotate_in_place']:  # if we are not rotating in place, make sure we have someplace to put the data
            measurement['post_processed_data'] = np.zeros(measurement[self.properties['data_to_pca']].shape)

        rotation = self.properties['pca_fit'].components_

        if self.properties['rotate_in_place']:
            measurement[self.properties['data_to_pca']][self.properties['demod_index']] = np.einsum('ijlk,mi->mjlk',
                                                                       measurement[self.properties['data_to_pca']][self.properties['demod_index']],
                                                                       rotation)
        else:
            measurement['post_processed_data'][self.properties['demod_index']] = np.einsum('ijlk,mi->mjlk',
                                                              copy.copy(measurement[self.properties['data_to_pca']][self.properties['demod_index']]),
                                                              rotation)

        measurement['Demod_Index_{}'.format(self.properties['demod_index'])]['Meas_PCA_Rotation'] = copy.deepcopy(self.properties)


class Meas_Fit_Sin_Exp(Measurement_Process):
    def __init__(self, data_to_fit='post_demod_data', demod_index=0, trigger=0, I_or_Q_axis=0, fit_sin=True, time_axis=None,
                 raw_fit={}, fit_data={}):

        super(Meas_Fit_Sin_Exp, self).__init__()
        self.properties['process_name'] = 'Meas_Fit_Sin_Exp'
        self.properties['data_to_fit'] = data_to_fit
        self.properties['trigger'] = trigger
        self.properties['I_or_Q_axis'] = I_or_Q_axis
        self.properties['fit_sin'] = fit_sin
        self.properties['time_axis'] = time_axis
        self.properties['raw_fit'] = raw_fit
        self.properties['demod_index'] = demod_index
        self.properties['fit_data'] = fit_data

    def fit(self, measurement=None, **kargs):

        if 'Demod_Index_{}'.format(self.properties['demod_index']) not in measurement.keys():
            measurement['Demod_Index_{}'.format(self.properties['demod_index'])] = dict()

        try:
            spacing = measurement['sequence']['element_spacing']
        except:
            warnings.warn("Meas_Fit_Sin_Exp failed, sequence not found in measurement, so time axis is indeterminate.")
            return  # if there is no specified spacing, return without fitting, better to error then return wrong data

        data_to_use = measurement[self.properties['data_to_fit']].mean(2) #Average over the repetitions
        data_to_use = data_to_use[self.properties['demod_index'],self.properties['I_or_Q_axis'],self.properties['trigger']] #select the appropriate trigger

        data_shape = data_to_use.shape
        t_axis = np.arange(0, data_shape[-1]) * spacing[us]

        curve_fit = fitTimeTrace(t_axis, data_to_use.T, fit_sin=self.properties['fit_sin'])

        self.properties['fit_data'] = residual_expsin(curve_fit, t_axis)
        self.properties['raw_fit'] = curve_fit
        self.properties['time_axis'] = t_axis

        measurement['Demod_Index_{}'.format(self.properties['demod_index'])]['Meas_Fit_Sin_Exp'] = copy.deepcopy(self.properties)


    def process(self, measurement, **kargs):
        # in this case they are the same thing
        self.fit(measurement)


class Meas_Herald_Threshold(Measurement_Process):
    def __init__(self, herald_threshold=0, iq_axis=0, trigger=0, data_to_herald='post_demod_data'):
        super(Meas_Herald_Threshold, self).__init__()

        self.properties['process_name'] = 'Meas_Herald_Threshold'
        self.properties['herald_threshold'] = herald_threshold
        self.properties['iq_axis'] = iq_axis
        self.properties['trigger'] = trigger
        self.properties['data_to_herald'] = data_to_herald

    def process(self, measurement, **kwargs):
        unheralded_data = measurement[self.properties['data_to_herald']][:, self.properties['iq_axis'], :, self.properties['trigger'], :]

        measurement_shape = measurement[self.properties['data_to_herald']].shape

        herald_shape = unheralded_data.shape

        herald = np.array(self.properties['herald_threshold'])

        if len(herald.shape) == 0:
            herald = herald * np.ones(unheralded_data.shape[0])

        herald = herald.reshape(-1, 1, 1)

        pass_herald = np.array(unheralded_data < herald)

        # The issue is now what do we do when we get differing numbers of measurements
        # which pass heralding, IE: we have a sequence of 128 length, which we have 100
        # captures of. Lets say of the 12800 total measurements, only 1 fails the herald check.
        # in this case we remove 127 elements, 1 from each of the other sequence element.
        # We then only return 128x99 instead of a mixed array of 128x(100 or 99).
        # this is beneficial both from a statistics point of view and programming, as everything
        # stays evenly spaced


        n_good_measurements = pass_herald.sum(1).min()

        # Now lets make the mask super
        for qubit in range(herald_shape[0]):
            for seq_element in range(herald_shape[2]):
                throw_away = sum(pass_herald[qubit, :, seq_element]) - n_good_measurements
                final_index = throw_away
                while np.sum(pass_herald[qubit, :final_index, seq_element]) < throw_away:
                    final_index += 1
                pass_herald[qubit, 0:final_index, seq_element] = False

        # return pass_herald
        tmp_shape = list(measurement_shape)

        tmp_shape[2] = n_good_measurements
        processed_data = np.zeros(tmp_shape)

        for channel in range(measurement_shape[1]):
            for trigger in range(measurement_shape[3]):
                processed_data[:, channel, :, trigger, :] = (
                    measurement[self.properties['data_to_herald']][:, channel, :, trigger, :].T[pass_herald.T].reshape(
                        np.array(processed_data.shape)[[(4, 2, 0)]])).T

        measurement['post_processed_data'] = processed_data
        measurement['Meas_Herald_Threshold'] = copy.deepcopy(self.properties)


class Meas_Fit_Gauss_MM(Measurement_Process):
    def __init__(self, n_gaussians=2, demod_index=0, data_to_fit='post_demod_data', trigger=0,
                 GMM_fit=None, means=None, std_deviations=None, **kwargs):
        super(Meas_Fit_Gauss_MM, self).__init__()

        self.properties['process_name'] = 'Meas_Fit_Gauss_MM'
        self.properties['n_gaussians'] = n_gaussians
        self.properties['demod_index'] = demod_index
        self.properties['data_to_fit'] = data_to_fit
        self.properties['trigger'] = trigger
        self.properties['GMM_fit'] = GMM_fit
        self.properties['std_deviations'] = std_deviations
        self.properties['means'] = means
        self.properties['classification'] = None

    def fit(self, measurement=None, **kargs):
        # [demod freq, IQ channel, repetition, trigger, sequence element]
        samples = measurement[self.properties['data_to_fit']][self.properties['demod_index'], :, :, self.properties['trigger'], :].reshape(2, -1)
        samples = samples.T

        if self.properties['n_gaussians'] is None:
            # If n_gaussians not found, calculated them based on the BIC criterion
            bics = []
            # Fit up to 7 gaussians and record the bics
            for i in range(1,8):
                gmix = mixture.GaussianMixture(n_components=i, covariance_type='full')
                gmix.fit(samples)
                bics.append(gmix.bic(samples))

            # attempt to estimate the number of gaussians using the BIC criterion
            # pick the point at which the the change in number of gaussians
            # results in the BIC entering one standard deviation from the mean BIC
            self.properties['n_gaussians'] = np.argmin(np.array(bics) > (3*np.std(bics[-3:])+np.mean(bics[-3:]))) + 1

            # record the size of our BICs for posterity
            self.properties['BICs'] = bics

        gmix = mixture.GaussianMixture(n_components=self.properties['n_gaussians'], covariance_type='full')
        gmix.fit(samples)

        sort_dir = np.argsort(gmix.means_[:, 0])
        gmix.means_ = gmix.means_[sort_dir]
        gmix.covariances_ = gmix.covariances_[sort_dir]

        std_deviations, rotations = np.linalg.eig(gmix.covariances_)

        self.properties['means'] = gmix.means_
        self.properties['std_deviations'] = std_deviations ** .5
        self.properties['GMM_fit'] = gmix

    def process(self, measurement, **kargs):

        if 'Demod_Index_{}'.format(self.properties['demod_index']) not in measurement.keys():
            measurement['Demod_Index_{}'.format(self.properties['demod_index'])] = dict()


        samples = measurement[self.properties['data_to_fit']][self.properties['demod_index'], :, :, self.properties['trigger'], :].reshape(2, -1)
        samples = samples.T

        sample_shape = measurement[self.properties['data_to_fit']][self.properties['demod_index'], :, :, self.properties['trigger'], :].shape

        self.properties['classification'] = self.properties['GMM_fit'].predict(samples).T.reshape(sample_shape[1:])

        measurement['Demod_Index_{}'.format(self.properties['demod_index'])]['Meas_Fit_Gauss_MM'] = copy.deepcopy(self.properties)



