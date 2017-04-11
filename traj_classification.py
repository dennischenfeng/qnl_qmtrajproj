
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import svm
from sklearn import preprocessing
from sklearn.decomposition import PCA
import time

class NaiveInt:
    """
    Classifier using the naive integration method.
    Assumes 4 labels only: gg, ge, eg, ee.
    Assumes all trajectories have same number of time bins.
    """

    def __init__(self):
        self.traj = None #the demodded traj that is input into the fit(.) method; indices: iqIndex, trajIndex, timeIndex
        self.doRotate = True #whether or not to rotate the integrated I,Q points in I,Q space such that all the info is contained in the I quadrature
        self.intTraj = None #integrated (aka take the mean) trajectories; indices: [label][iqIndex, trajIndex]
        self.hist2d_gg = None #hist2d of the gg points
        self.hist2d_exc = None #hist2d of the exc points (exc means {ge,eg,ee})
        self.binEdges_I = None #bin edges in the hist2d's above
        self.binEdges_Q = None
        self.theta = 0 #angle to rotate the I,Q points in I,Q space; ensures that gg blob is to the left of the exc blob
        self.intTraj_rotated = None # "_rotated" means after the theta rotation
        self.hist2d_gg_rotated = None
        self.hist2d_exc_rotated = None
        self.binEdges_I_rotated = None
        self.binEdges_Q_rotated = None
        self.decBound_ggexc_I = 0 # the I quadrature decision boundary (i.e. in units of I) to classify either gg or exc
        self.trainFid_ggexc_I = 0 # the fidelity of the decBound on the training data


    def fit(self, traj, labels, suppressPlots=False, doRotate=True, numBins=70):
        """
        Fits the NaiveInt classifier given demodded traj's and their corresponding labels. These compose the training set.
        Also can plot some graphs to let the user ensure the fit fitted correctly.

        Params:
        traj - the demodded trajectories in an np array, with 3 indices: iqIndex, trajIndex, timeIndex
        labels - the labels corresponding to the trajectories, 1d np array
        suppressPlots - False if you want the plots to display (displaying the plot pauses the program)
        doRotate - True if you want this method to rotate the data in IQ space so that all the info is in the I quadrature
        numBins - number of bins (in both the I and Q directions) of the histograms

        Returns:
        self
        """

        #: Integrate the I,Q trajs --> intTraj
        ###

        assert traj.shape[1] == labels.shape[0], "Error: numTraj from 'traj' isn't the same as from 'labels'."

        self.traj = traj
        self.doRotate = doRotate
        numTotalTraj = traj.shape[1]
        numTimeBins = traj.shape[2]  # 5000 #num time bins per traj

        numTraj = [0, 0, 0, 0]  #num trajectories for each label; indices: labelIndex (aka label)
        traj_separated = [[], [], [], []] # trajectories separated into 4 groups based on label; indices: [label][trajIndex][iqIndex, timeIndex]

        for trajIndex in np.arange(numTotalTraj):
            label = int(labels[trajIndex])
            numTraj[label] = numTraj[label] + 1
            traj_separated[label] = traj_separated[label] + [traj[:, trajIndex, :]]

        self.intTraj = [np.zeros([2, numTraj[0]]), np.zeros([2, numTraj[1]]), np.zeros([2, numTraj[2]]), np.zeros([2, numTraj[3]])] #indices: [label][iqIndex, trajIndex]

        for label in [0, 1, 2, 3]:
            for trajIndex in np.arange(numTraj[label]):
                self.intTraj[label][0, trajIndex] = traj_separated[label][trajIndex][0, :].mean() / numTimeBins
                self.intTraj[label][1, trajIndex] = traj_separated[label][trajIndex][1, :].mean() / numTimeBins



        #: calculate hist2d's and plot them
        ###

        minIntTraj_I = np.min([np.min(self.intTraj[0][0, :]), np.min(self.intTraj[1][0, :]), np.min(self.intTraj[2][0, :]), np.min(self.intTraj[3][0, :])])
        maxIntTraj_I = np.max([np.max(self.intTraj[0][0, :]), np.max(self.intTraj[1][0, :]), np.max(self.intTraj[2][0, :]), np.max(self.intTraj[3][0, :])])
        minIntTraj_Q = np.min([np.min(self.intTraj[0][1, :]), np.min(self.intTraj[1][1, :]), np.min(self.intTraj[2][1, :]), np.min(self.intTraj[3][1, :])])
        maxIntTraj_Q = np.max([np.max(self.intTraj[0][1, :]), np.max(self.intTraj[1][1, :]), np.max(self.intTraj[2][1, :]), np.max(self.intTraj[3][1, :])])
        self.hist2d_gg, self.binEdges_I, self.binEdges_Q = np.histogram2d(self.intTraj[0][0, :], self.intTraj[0][1, :], bins=numBins, range=[[minIntTraj_I, maxIntTraj_I], [minIntTraj_Q, maxIntTraj_Q]])

        intTraj_exc_I = np.concatenate([self.intTraj[1][0, :], self.intTraj[2][0, :], self.intTraj[3][0, :]])
        intTraj_exc_Q = np.concatenate([self.intTraj[1][1, :], self.intTraj[2][1, :], self.intTraj[3][1, :]])
        self.hist2d_exc, _ , _ = np.histogram2d(intTraj_exc_I, intTraj_exc_Q, bins=numBins, range=[[minIntTraj_I, maxIntTraj_I], [minIntTraj_Q, maxIntTraj_Q]])

        #: plot the (non-rotated) plots
        if suppressPlots == False:
            coordsX, coordsY = np.meshgrid(self.binEdges_I, self.binEdges_Q)
            fig, axarr = plt.subplots(2, 3)

            axarr[0, 0].pcolormesh(coordsX, coordsY, self.hist2d_gg.T + self.hist2d_exc.T, cmap='jet')
            axarr[0, 0].set_title('hist2d of gg and exc')

            axarr[0, 1].plot(self.binEdges_I[:-1], self.hist2d_gg.sum(1))
            axarr[0, 1].plot(self.binEdges_I[:-1], self.hist2d_exc.sum(1))
            axarr[0, 1].set_title('Integrated I')

            axarr[0, 2].plot(self.binEdges_Q[:-1], self.hist2d_gg.sum(0))
            axarr[0, 2].plot(self.binEdges_Q[:-1], self.hist2d_exc.sum(0))
            axarr[0, 2].set_title('Integrated Q')



        #: calculate rotated hist2d's and plot them
        ###

        if doRotate:
            self.theta = np.arctan2(intTraj_exc_I.mean() - self.intTraj[0][0, :].mean(), intTraj_exc_Q.mean() - self.intTraj[0][1, :].mean())
            self.intTraj_rotated = [np.zeros([2, numTraj[0]]), np.zeros([2, numTraj[1]]), np.zeros([2, numTraj[2]]), np.zeros([2, numTraj[3]])] #indices: [label][iqIndex, trajIndex]

            for label in [0, 1, 2, 3]:
                mag = np.sqrt(self.intTraj[label][0,:]**2 + self.intTraj[label][1,:]**2) #the magnitude (length) of each IQ vector in I-Q space
                phi = np.arctan2(self.intTraj[label][0,:], self.intTraj[label][1,:]) #the phase (i.e. angle) of each IQ vector in IQ space
                self.intTraj_rotated[label][0,:] = mag * np.cos(phi - self.theta)
                self.intTraj_rotated[label][1,:] = mag * np.sin(phi - self.theta)

            #: calculate 2d hists (rotated)
            minIntTraj_I_rotated = np.min([np.min(self.intTraj_rotated[0][0, :]), np.min(self.intTraj_rotated[1][0, :]), np.min(self.intTraj_rotated[2][0, :]), np.min(self.intTraj_rotated[3][0, :])])
            maxIntTraj_I_rotated = np.max([np.max(self.intTraj_rotated[0][0, :]), np.max(self.intTraj_rotated[1][0, :]), np.max(self.intTraj_rotated[2][0, :]), np.max(self.intTraj_rotated[3][0, :])])
            minIntTraj_Q_rotated = np.min([np.min(self.intTraj_rotated[0][1, :]), np.min(self.intTraj_rotated[1][1, :]), np.min(self.intTraj_rotated[2][1, :]), np.min(self.intTraj_rotated[3][1, :])])
            maxIntTraj_Q_rotated = np.max([np.max(self.intTraj_rotated[0][1, :]), np.max(self.intTraj_rotated[1][1, :]), np.max(self.intTraj_rotated[2][1, :]), np.max(self.intTraj_rotated[3][1, :])])
            self.hist2d_gg_rotated, self.binEdges_I_rotated, self.binEdges_Q_rotated = np.histogram2d(self.intTraj_rotated[0][0, :], self.intTraj_rotated[0][1, :], bins=numBins, range=[[minIntTraj_I_rotated, maxIntTraj_I_rotated], [minIntTraj_Q_rotated, maxIntTraj_Q_rotated]])

            intTraj_exc_I_rotated = np.concatenate([self.intTraj_rotated[1][0, :], self.intTraj_rotated[2][0, :], self.intTraj_rotated[3][0, :]])
            intTraj_exc_Q_rotated = np.concatenate([self.intTraj_rotated[1][1, :], self.intTraj_rotated[2][1, :], self.intTraj_rotated[3][1, :]])
            self.hist2d_exc_rotated, _ , _ = np.histogram2d(intTraj_exc_I_rotated, intTraj_exc_Q_rotated, bins=numBins, range=[[minIntTraj_I_rotated, maxIntTraj_I_rotated], [minIntTraj_Q_rotated, maxIntTraj_Q_rotated]])


            #: plot rotated plots
            if suppressPlots == False:
                coordsX_rotated, coordsY_rotated = np.meshgrid(self.binEdges_I_rotated, self.binEdges_Q_rotated)

                axarr[1, 0].pcolormesh(coordsX_rotated, coordsY_rotated, self.hist2d_gg_rotated.T + self.hist2d_exc_rotated.T, cmap='jet')
                axarr[1, 0].set_title('hist2d of gg and exc, rotated')

                axarr[1, 1].plot(self.binEdges_I_rotated[:-1], self.hist2d_gg_rotated.sum(1))
                axarr[1, 1].plot(self.binEdges_I_rotated[:-1], self.hist2d_exc_rotated.sum(1))
                axarr[1, 1].set_title('Integrated I, rotated')

                axarr[1, 2].plot(self.binEdges_Q_rotated[:-1], self.hist2d_gg_rotated.sum(0))
                axarr[1, 2].plot(self.binEdges_Q_rotated[:-1], self.hist2d_exc_rotated.sum(0))
                axarr[1, 2].set_title('Integrated Q, rotated')



        #: find decBound and trainFid, show plots
        ###

        if doRotate:
            hist2d_gg_chosen = self.hist2d_gg_rotated
            hist2d_exc_chosen = self.hist2d_exc_rotated
            binEdges_I_chosen = self.binEdges_I_rotated
        else:
            hist2d_gg_chosen = self.hist2d_gg
            hist2d_exc_chosen = self.hist2d_exc
            binEdges_I_chosen = self.binEdges_I

        temp_I=[] # temp variable to help calculate the fidelity; it's regarding I quadrature
        num_gg = hist2d_gg_chosen.sum(1).sum() #number of gg points
        num_exc = hist2d_exc_chosen.sum(1).sum() #number of exc points

        for k in np.arange(numBins+1):
            temp_I.append((hist2d_gg_chosen.sum(1)[0:k]/num_gg - hist2d_exc_chosen.sum(1)[0:k]/num_exc).sum())
        self.decBound_ggexc_I = binEdges_I_chosen[np.argmax(temp_I)] # decision boundary
        self.trainFid_ggexc_I = (1 + np.max(temp_I)) / 2   #full fid for I quadrature, using naive integration

        print 'trainFid_ggexc_I: ', self.trainFid_ggexc_I
        print 'decBound_ggexc_I: ', self.decBound_ggexc_I

        if suppressPlots == False:
            plt.show()


        return self


    def score(self, traj, labels):
        """
        Calculates fidelity on the given labelled test set.

        Params:
        traj - the demodded trajectories in an np array, with 3 indices: iqIndex, trajIndex, timeIndex
        labels - the labels corresponding to the trajectories, 1d np array

        Returns:
        fidelity
        """

        numTotalTraj = traj.shape[1]
        numTimeBins = traj.shape[2]

        num_gg_exc = 0 # num of samples that are predicted 'gg' but are actually exc
        num_exc_gg = 0  # num of samples that are predicted 'exc' but are actually gg
        num_exc = 0 # total num of true exc sample
        num_gg = 0  # total num of true exc sample

        #: get intTraj_rotated
        ###

        intTraj = np.zeros([2, numTotalTraj]) # indices: iqIndex, trajIndex
        for trajIndex in np.arange(numTotalTraj):
            intTraj[0, trajIndex] = traj[0, trajIndex, :].mean() / numTimeBins
            intTraj[1, trajIndex] = traj[1, trajIndex, :].mean() / numTimeBins

        intTraj_rotated = np.zeros([2, numTotalTraj]) #indices: iqIndex, trajIndex
        mag = np.sqrt(intTraj[0,:]**2 + intTraj[1,:]**2) #the magnitude (length) of each IQ vector in I-Q space
        phi = np.arctan2(intTraj[0,:], intTraj[1,:]) #the phase (i.e. angle) of each IQ vector in IQ space
        intTraj_rotated[0,:] = mag * np.cos(phi - self.theta)
        intTraj_rotated[1,:] = mag * np.sin(phi - self.theta)

        #: calculate fid
        ###

        for i in np.arange(numTotalTraj):
            if labels[i] in [1, 2, 3]:
                num_exc = num_exc + 1
                if intTraj_rotated[0, i] <= self.decBound_ggexc_I:
                    num_gg_exc = num_gg_exc + 1
            else:
                num_gg = num_gg + 1
                if intTraj_rotated[0, i] > self.decBound_ggexc_I:
                    num_exc_gg = num_exc_gg + 1

        prob_gg_exc = 1.0 * num_gg_exc / num_exc
        prob_exc_gg = 1.0 * num_exc_gg / num_gg

        fid_ggexc = 1 - (prob_gg_exc + prob_exc_gg) / 2

        return fid_ggexc


    def predict(self, traj):
        print 'needs implementation'


class SWInt_DiffAvgTraj:
    """
    Classifier using the slot weights method. Calculates slot weights using the difference of average trajectories.
    Assumes 4 labels only: gg, ge, eg, ee.
    Assumes all trajectories have same number of time bins.
    """


class SWInt_SVM:
    """
    Classifier using the slot weights method. Calculates (hopefully optimal) slot weights using an SVM. Includes option to tune the hyperparameter C for the linear SVM.
    Assumes 4 labels only: gg, ge, eg, ee.
    Assumes all trajectories have same number of time bins.
    """

    def __init__(self, typeSVM='linear', slotSize=100, usePCA=False, useExtraFeatures=True):
        self.clf_SVM = None #svm.LinearSVC(C=C)
        self.typeSVM = typeSVM
        self.slotSize = slotSize
        self.usePCA = usePCA
        self.pca = None
        self.useExtraFeatures = useExtraFeatures

        validTypeSVMs = ['linear', 'rbf', 'linear_v2']
        if self.typeSVM not in validTypeSVMs:
            print "Error: typeSVM is not valid."

    def __formatIntoFeatureVectorsAndLabels(self, traj, labels, fitNewPCA=False):
        """
        Helper function. Converts the traj and labels into a feature vector matrix (design matrix) and a labels array, which is the proper format to input into the SVM.
        :param traj: the demodded trajectories in an np array, with 3 indices: iqIndex, trajIndex, timeIndex
        :param labels: the labels corresponding to the trajectories, 1d np array
        :return: scaled (i.e. mean removal and scaled variance to 1) inputVectors (each feature vector is the I vector concatted with the Q vector); and labels_ggexc (labels for each trajectory, 0 for gg, 1 for exc)
        """
        numTotalTraj = traj.shape[1]
        numTimeBins = traj.shape[2]  # 5000 #num time bins per traj

        #: get inputVectors and labels_ggexc
        ###
        numSlots = numTimeBins / self.slotSize  # num slots per traj

        traj_slotted = np.reshape(traj, (2, numTotalTraj, numSlots, self.slotSize)).mean(3)

        if self.useExtraFeatures:
            # derivative features
            traj_slotted_derivative = np.gradient(traj_slotted, axis=2)
            inputVectors = np.concatenate((traj_slotted[0, :, :], traj_slotted[1, :, :], traj_slotted_derivative[0, :, :], traj_slotted_derivative[1, :, :]), axis=1)  # sample vectors to input into the SVM;

            # fft features
            traj_slotted_fft = np.fft.fft(traj_slotted)
            inputVectors = np.concatenate((inputVectors[:, :], traj_slotted_fft[0, :, :], traj_slotted_fft[1, :, :]), axis=1)  # sample vectors to input into the SVM;
        else:
            inputVectors = np.concatenate((traj_slotted[0, :, :], traj_slotted[1, :, :]), axis=1)  # sample vectors to input into the SVM;

        labels_ggexc = np.array([0 if labels[i]==0 else 1 for i in np.arange(numTotalTraj) ]) # this groups gg as label 0, and exc as label 1

        inputVectors = preprocessing.scale(inputVectors)

        if self.usePCA:
            if fitNewPCA:
                self.pca = PCA(n_components=20, whiten=True)
                self.pca.fit(inputVectors)
                print 'PCA explained variance ratio: ', self.pca.explained_variance_ratio_

            inputVectors = self.pca.transform(inputVectors)

        return inputVectors, labels_ggexc


    def __findFidelity(self, inputVectors, labels_ggexc):
        """
        Helper function. Calculates the fidelity (fid_ggexc)
        :param inputVectors:
        :param labels_ggexc:
        :return: fidelity
        """
        numTotalTraj = inputVectors.shape[0]

        #: find train fidelity
        ###
        labels_ggexc_predicted = self.clf_SVM.predict(inputVectors)

        num_gg_exc = 0  # num of samples that are predicted 'gg' but are actually exc
        num_exc_gg = 0  # num of samples that are predicted 'exc' but are actually gg
        num_exc = 0  # total num of true exc sample
        num_gg = 0  # total num of true exc sample

        for i in np.arange(numTotalTraj):
            if labels_ggexc[i] == 1:
                num_exc = num_exc + 1
                if labels_ggexc_predicted[i] == 0:
                    num_gg_exc = num_gg_exc + 1
            else:
                num_gg = num_gg + 1
                if labels_ggexc_predicted[i] == 1:
                    num_exc_gg = num_exc_gg + 1

        prob_gg_exc = 1.0 * num_gg_exc / num_exc #probability of predicting gg, given that the state is exc
        prob_exc_gg = 1.0 * num_exc_gg / num_gg

        fid_ggexc = 1 - (prob_gg_exc + prob_exc_gg) / 2

        return fid_ggexc


    def fit(self, traj, labels, tuneC=True, lstC=None, validationFraction=0.2, suppressPlots=False):
        """
        Fits the SVM.

        :param traj:
        :param labels:
        :param slotSize: size (number of time units) of each slot, for performing slot weights integration
        :param tuneC: True if you want the method to tune C
        :param lstC: None if you want to use the default list of C's to sweep through for finding the optimal C; should be an np array or a list.
        :param validationFraction: fraction of the traj's to use as the validation set (tuning C happens on the validation set)
        :return: self
        """
        numTotalTraj = traj.shape[1]

        inputVectors, labels_ggexc = self.__formatIntoFeatureVectorsAndLabels(traj, labels, fitNewPCA=self.usePCA)

        #: fit the clf_SVM, tune C if chosen to
        ###
        inputVectors_shuffled, labels_ggexc_shuffled = shuffle(inputVectors, labels_ggexc)

        if tuneC == False:
            if self.typeSVM == 'linear':
                self.clf_SVM = svm.LinearSVC(C=1.3e-4)
            elif self.typeSVM == 'rbf':
                self.clf_SVM = svm.SVC(kernel='rbf', C=0.65) #C=1.32
            elif self.typeSVM == 'linear_v2':
                self.clf_SVM = svm.SVC(kernel='linear', C=1.3e-4)

            self.clf_SVM.fit(inputVectors_shuffled, labels_ggexc_shuffled)
        else:
            #: tune C
            print 'Tuning C...'
            lstFid = []
            startIndex_validation = int(numTotalTraj * (1 - validationFraction))

            if lstC is None:
                if self.typeSVM == 'linear':
                    lstC = 10**np.linspace(-5, -2, 20)
                elif self.typeSVM == 'rbf':
                    lstC = 10 ** np.linspace(-1, 1, 10)
                elif self.typeSVM == 'linear_v2':
                    lstC = 10 ** np.linspace(-7, 0.5, 30)

            for C in lstC:
                if self.typeSVM == 'linear':
                    self.clf_SVM = svm.LinearSVC(C=C)
                elif self.typeSVM == 'rbf':
                    self.clf_SVM = svm.SVC(kernel='rbf', C=C)
                elif self.typeSVM == 'linear_v2':
                    self.clf_SVM = svm.SVC(kernel='linear', C=C)

                self.clf_SVM.fit(inputVectors_shuffled[0:startIndex_validation],
                                 labels_ggexc_shuffled[0:startIndex_validation])
                temp_fid = self.__findFidelity(inputVectors_shuffled[startIndex_validation:],
                                               labels_ggexc_shuffled[startIndex_validation:])
                print 'On C =', C, '; validation fid =', temp_fid
                lstFid = lstFid + [temp_fid]

            plt.figure()
            plt.plot(lstC, lstFid)
            plt.gca().set_xscale('log')
            plt.title('Tuning C')
            plt.xlabel('C')
            plt.ylabel('Fidelity')

            optimalC = lstC[np.argmax(lstFid)]
            print 'Chose optimal C = ', optimalC


            # #: tune gamma
            # if typeSVM == 'rbf':
            #     print 'Tuning gamma...'
            #     lstFid = []
            #     startIndex_validation = int(numTotalTraj * (1 - validationFraction))
            #
            #     lstGamma = [0.00088]
            #     # lstGamma = 10**np.linspace(-5.5, -2, 20)
            #
            #     for gamma in lstGamma:
            #         self.clf_SVM = svm.SVC(kernel='rbf', C=optimalC, gamma=gamma)
            #
            #         self.clf_SVM.fit(inputVectors_shuffled[0:startIndex_validation],
            #                          labels_ggexc_shuffled[0:startIndex_validation])
            #         temp_fid = self.__findFidelity(inputVectors_shuffled[startIndex_validation:],
            #                                        labels_ggexc_shuffled[startIndex_validation:])
            #         print 'On gamma =', gamma, '; validation fid =', temp_fid
            #         lstFid = lstFid + [temp_fid]
            #
            #     plt.figure()
            #     plt.plot(lstGamma, lstFid)
            #     plt.gca().set_xscale('log')
            #     plt.title('Tuning gamma')
            #     plt.xlabel('gamma')
            #     plt.ylabel('Fidelity')
            #
            #     optimalGamma = lstGamma[np.argmax(lstFid)]
            #     print 'Chose optimal gamma = ', optimalGamma



            if self.typeSVM == 'linear':
                self.clf_SVM = svm.LinearSVC(C=optimalC)
            elif self.typeSVM == 'rbf':
                # self.clf_SVM = svm.SVC(kernel='rbf', C=optimalC, gamma=optimalGamma)
                self.clf_SVM = svm.SVC(kernel='rbf', C=optimalC)
            elif self.typeSVM == 'linear_v2':
                self.clf_SVM = svm.SVC(kernel='linear', C=optimalC)

            self.clf_SVM.fit(inputVectors_shuffled, labels_ggexc_shuffled)




        fid_ggexc = self.__findFidelity(inputVectors_shuffled, labels_ggexc_shuffled)

        # print 'train fid: ', fid_ggexc

        if suppressPlots == False:
            plt.show()

        return self


    def score(self, traj, labels):
        """
        Calculates the fidelity.

        :param traj:
        :param labels:
        :return: fidelity
        """


        inputVectors, labels_ggexc = self.__formatIntoFeatureVectorsAndLabels(traj, labels)
        fid_ggexc = self.__findFidelity(inputVectors, labels_ggexc)

        return fid_ggexc

    def predict(self, traj):
        """
        Uses the SVM to predict the labels of the input demodded trajectories.

        :param traj:
        :return: 1D np array of the predicted labels
        """
        numTotalTraj = traj.shape[1]

        inputVectors, _ = self.__formatIntoFeatureVectorsAndLabels(traj, np.zeros(numTotalTraj)) #puts in a fake labels set (because we don't care about the labels)

        labels_ggexc_predicted = self.clf_SVM.predict(inputVectors)

        return labels_ggexc_predicted





def demod(rawdata):
    """
    Performs demodulation on the raw trajectories.

    :param rawdata: acquired raw data; should be an np.array with shape (2, numTrajPerLabel, 4, numTimeBinsPerTrajectory)
    :return: a tuple: (traj, labels); traj is the demodded trajectories (which can be directly inputted into the classifiers), labels is the label for each trajectory
    """

    #: params for the rawdata
    numTrajs = rawdata.shape[1] ##num trajectories per label
    numTimeBins = rawdata.shape[3] ##number of time bins in each trajectory

    #: params for demod
    demod_freq =   0.047000000 #47 MHz = 0.047GHz
    clock_freq = 1.000000000 # 1000MHz =1 GHz
    rotation = 0
    # demod_decay = None  #obsolete
    start_window = 0 #units of time
    end_window = numTimeBins

    #: demod code
    times = np.arange( int(start_window*clock_freq), int(end_window*clock_freq) ) * float(demod_freq/(clock_freq)) #unitless
    demod_exp = np.exp(1j*(2*np.pi*times-rotation))
    demod_exp_I = np.array(np.real(demod_exp), dtype='float32')
    demod_exp_Q = np.array(np.imag(demod_exp), dtype='float32')

    #: traj_demod and traj_av_demod
    traj_demod=np.zeros((2, 4, numTrajs, numTimeBins)) # indices are: iqIndex, labelIndex, trajIndex, timeIndex  ;  value is the i or q quadrature value
    traj_av_demod=np.zeros((2, 4, numTimeBins)) # averages over the trajs; indices: iqIndex, labelIndex, timeIndex


    for labelIndex in np.arange(4):
        for trajIndex in np.arange(numTrajs):
            traj_demod[0,labelIndex,trajIndex]=( (rawdata[0, trajIndex, labelIndex, :] - rawdata[0, trajIndex, labelIndex, :].mean())*demod_exp_I    -    (rawdata[1, trajIndex, labelIndex, :] - rawdata[1, trajIndex, labelIndex, :].mean())*demod_exp_Q)
            traj_demod[1,labelIndex,trajIndex]=( (rawdata[0, trajIndex, labelIndex, :] - rawdata[0, trajIndex, labelIndex, :].mean())*demod_exp_Q    +    (rawdata[1, trajIndex, labelIndex, :] - rawdata[1, trajIndex, labelIndex, :].mean())*demod_exp_I)
        traj_av_demod[0, labelIndex] = traj_demod[0, labelIndex,:,:].mean(0)
        traj_av_demod[1, labelIndex] = traj_demod[1, labelIndex,:,:].mean(0)

    #: preprocess the data a little
    traj = np.concatenate((traj_demod[:,0,:,:], traj_demod[:,1,:,:], traj_demod[:,2,:,:], traj_demod[:,3,:,:]), axis=1)
    labels = np.concatenate((np.zeros(numTrajs), np.ones(numTrajs), 2*np.ones(numTrajs), 3*np.ones(numTrajs)), axis=0)

    return traj, labels