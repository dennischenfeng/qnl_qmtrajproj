
import numpy as np
import matplotlib.pyplot as plt




class NaiveInt:
    """Classifier using the naive integration method.
    Assumes 4 labels only: gg, ge, eg, ee."""

    def __init__(self):
        self.traj = None
        self.doRotate = True
        self.intTraj = None
        self.hist2d_gg = None
        self.hist2d_exc = None
        self.binEdges_I = None
        self.binEdges_Q = None
        self.theta = 0
        self.intTraj_rotated = None
        self.hist2d_gg_rotated = None
        self.hist2d_exc_rotated = None
        self.binEdges_I_rotated = None
        self.binEdges_Q_rotated = None
        self.decBound_ggexc_I = 0
        self.trainFid_ggexc_I = 0

    def fit(self, traj, labels, suppressPlots=False, doRotate=True, numBins=70):
        """Fits the NaiveInt classifier given demodded traj's and their corresponding labels. These compose the training set.
        Also can plot some graphs to let the user ensure the fit fitted correctly."""

        #: Integrate the I,Q trajs --> intTraj

        assert traj.shape[1] == labels.shape[0], "Error: numTraj from traj isn't the same as from labels."

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
                self.intTraj[label][0, trajIndex] = traj_separated[label][trajIndex][0, :].mean()
                self.intTraj[label][1, trajIndex] = traj_separated[label][trajIndex][1, :].mean()



        #: calculate hist2d's

        minIntTraj_I = np.min([np.min(self.intTraj[0][0, :]), np.min(self.intTraj[1][0, :]), np.min(self.intTraj[2][0, :]), np.min(self.intTraj[3][0, :])])
        maxIntTraj_I = np.max([np.max(self.intTraj[0][0, :]), np.max(self.intTraj[1][0, :]), np.max(self.intTraj[2][0, :]), np.max(self.intTraj[3][0, :])])
        minIntTraj_Q = np.min([np.min(self.intTraj[0][1, :]), np.min(self.intTraj[1][1, :]), np.min(self.intTraj[2][1, :]), np.min(self.intTraj[3][1, :])])
        maxIntTraj_Q = np.max([np.max(self.intTraj[0][1, :]), np.max(self.intTraj[1][0, :]), np.max(self.intTraj[2][1, :]), np.max(self.intTraj[3][1, :])])
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

            fig.subplots_adjust(hspace=0)
            plt.show()


        #: if doRotate is true, then calculate the rotated histograms

        if doRotate:
            self.theta = np.arctan2(intTraj_exc_I.mean() - self.intTraj[0][0, :].mean(), intTraj_exc_Q.mean() - self.intTraj[0][1, :].mean())
            self.intTraj_rotated = [np.zeros([2, numTraj[0]]), np.zeros([2, numTraj[1]]), np.zeros([2, numTraj[2]]), np.zeros([2, numTraj[3]])] #indices: [label][iqIndex, trajIndex]

            for label in [0, 1, 2, 3]:
                mag = np.sqrt(self.intTraj[label][0,:]**2 + self.intTraj[label][1,:]**2) #the the magnitude (length) of each IQ vector in I-Q space
                phi = np.arctan2(self.intTraj[label][0,:], self.intTraj[label][1,:])
                self.intTraj_rotated[label][0,:] = mag * np.cos(phi - theta)
                self.intTraj_rotated[label][1,:] = mag * np.sin(phi - theta)

        # @@@ plot 2d hists












