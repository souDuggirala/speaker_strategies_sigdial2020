import os
import numpy as np
from scipy.special import softmax, expit
import gc

DEVICE = 'cpu'

calibrations_path = '/common/users/ssd122/speaker_strategies_storage/compositional_model/logs/E006_calibrations'
teacher_phi_path = os.path.join(calibrations_path, '99th_percentile_5/calibrated_phis_99percentile.npy')
unnormalized_phis_path = os.path.join(calibrations_path, '99th_percentile_5/unnormalized_phis.npy')

teacher_phi = np.load(teacher_phi_path)
teacher_phi_max_across_colorspace = teacher_phi.max(axis=0)

#print(teacher_phi_max_across_colorspace.mean())
print(teacher_phi_max_across_colorspace.mean(axis = 0))
print(teacher_phi_max_across_colorspace.mean(axis = 1))
#print(teacher_phi_max_across_colorspace)

del teacher_phi
gc.collect()