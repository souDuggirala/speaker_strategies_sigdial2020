import os
import numpy as np
from scipy.special import softmax, expit
import gc

DEVICE = 'cpu'

calibrations_path = '/common/users/ssd122/speaker_strategies_storage/compositional_model/logs/E006_calibrations'
teacher_phi_path = os.path.join(calibrations_path, '99th_percentile_6/calibrated_phis_99percentile.npy')
unnormalized_phis_path = os.path.join(calibrations_path, '99th_percentile_6/unnormalized_phis.npy')

teacher_phi = np.load(teacher_phi_path)
teacher_phi_max_across_colorspace = teacher_phi.max(axis=2)

#print(teacher_phi_max_across_colorspace.mean()) #0.9136807663076983
print(teacher_phi_max_across_colorspace.mean(axis = 0)) #[0.94477314 0.79418259 0.90538586 0.95608992 0.94114984 0.94050324]
#print(teacher_phi_max_across_colorspace)

del teacher_phi
gc.collect()