
import numpy as np
import matplotlib.pyplot as plt
from traj_classification import *
import h5py


# for filename in ['../163223_Trajectories.hdf5']:
#     f = h5py.File(filename,'r')
#     rawdata = f['data']['Dependent0'][()][:, :, 0:4, :] #this turns the h5data into a numpy file
#     print filename
#     print 'Shape of raw data: ', rawdata.shape
#     f.close()
#
# #: params for the rawdata
# numTrajs = rawdata.shape[1] #5000 #num trajectories per label
# duration = rawdata.shape[3] #5000 #time duration of entire track, in nanoseconds
#
# #: params for demod
# demod_freq =   0.047000000 #47 MHz = 0.047GHz
# clock_freq = 1.000000000 # 1000MHz =1 GHz
# rotation = 0
# # demod_decay = None  #obsolete
# start_window = 0 #units of time
# end_window = duration #is this the correct end_window?
#
# #: demod code
# times = np.arange( int(start_window*clock_freq), int(end_window*clock_freq) ) * float(demod_freq/(clock_freq)) #unitless
# demod_exp = np.exp(1j*(2*np.pi*times-rotation))
# # if demod_decay is not None:   #obsolete
# #     demod_exp *= np.exp(-1*times/(demod_decay*demod_freq))
# demod_exp_I = np.array(np.real(demod_exp), dtype='float32')
# demod_exp_Q = np.array(np.imag(demod_exp), dtype='float32')
#
# #: traj_demod and traj_av_demod
# traj_demod=np.zeros((2, 4, numTrajs, duration)) # indices are: iqIndex, labelIndex, trajIndex, timeIndex  ;  value is the i or q quadrature value
# traj_av_demod=np.zeros((2, 4, duration)) # averages over the trajs; indices: iqIndex, labelIndex, timeIndex
#
#
# for labelIndex in np.arange(4):
#     for trajIndex in np.arange(numTrajs):
#         traj_demod[0,labelIndex,trajIndex]=( (rawdata[0, trajIndex, labelIndex, :] - rawdata[0, trajIndex, labelIndex, :].mean())*demod_exp_I    -    (rawdata[1, trajIndex, labelIndex, :] - rawdata[1, trajIndex, labelIndex, :].mean())*demod_exp_Q)
#         traj_demod[1,labelIndex,trajIndex]=( (rawdata[0, trajIndex, labelIndex, :] - rawdata[0, trajIndex, labelIndex, :].mean())*demod_exp_Q    +    (rawdata[1, trajIndex, labelIndex, :] - rawdata[1, trajIndex, labelIndex, :].mean())*demod_exp_I)
#     traj_av_demod[0, labelIndex] = traj_demod[0, labelIndex,:,:].mean(0)
#     traj_av_demod[1, labelIndex] = traj_demod[1, labelIndex,:,:].mean(0)
#
# np.save('traj_demod.npy', traj_demod)





# traj_demod = np.load('traj_demod.npy')
# print 'data loaded'
#
#
# #: preprocess the data a little
# traj = np.concatenate((traj_demod[:,0,:4000,:], traj_demod[:,1,:4000,:], traj_demod[:,2,:4000,:], traj_demod[:,3,:4000,:]), axis=1)
# labels = np.concatenate((np.zeros(4000), np.ones(4000), 2*np.ones(4000), 3*np.ones(4000)), axis=0)
# print 'traj and labels calculated'
# np.save('traj_train.npy', traj)
# np.save('labels_train.npy', labels)
#
# traj = np.concatenate((traj_demod[:,0,4000:,:], traj_demod[:,1,4000:,:], traj_demod[:,2,4000:,:], traj_demod[:,3,4000:,:]), axis=1)
# labels = np.concatenate((np.zeros(1000), np.ones(1000), 2*np.ones(1000), 3*np.ones(1000)), axis=0)
# print 'traj and labels calculated'
# np.save('traj_test.npy', traj)
# np.save('labels_test.npy', labels)



traj_train = np.load('traj_train.npy')
labels_train = np.load('labels_train.npy')
print 'train traj and labels loaded'

traj_test = np.load('traj_test.npy')
labels_test = np.load('labels_test.npy')
print 'test traj and labels loaded'


clf = NaiveInt()
clf.fit(traj_train, labels_train, doRotate=True, suppressPlots=False)
print 'test fid:', clf.score(traj_test, labels_test)

clf = SWInt_SVM()
clf.fit(traj_train, labels_train)
print 'test fid: ', clf.score(traj_test, labels_test)
