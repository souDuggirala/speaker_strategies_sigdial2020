import os
import pathlib
import yaml

FLOATX = 'float32'

'''
HERE = pathlib.Path(__file__).absolute().parents[0]
REPO_ROOT = HERE.parents[1]
DATA_ROOT = str(REPO_ROOT / "data")
'''

# Soumya - temporary solution to put correct path for my system,
    # need path of source code and not egg installation, because data and models are in source code
REPO_ROOT = pathlib.Path("/Users/soumyadugg/speaker_strategies_sigdial2020")
DATA_ROOT = str(REPO_ROOT / "data")

# XKCD dataset files
XKCD_DATASET_FILES = {
    "color_values": os.path.join(DATA_ROOT, "xkcd/color_values.npy"),
    "annotations": os.path.join(DATA_ROOT, "xkcd/annotations.csv")
}
XKCD_VOCAB = os.path.join(DATA_ROOT, "xkcd/vocab.json")

# Colors in Context dataset files
CIC_DATA_CSV = os.path.join(DATA_ROOT, "filteredCorpus.csv")
CIC_VECTORIZED_CSV = os.path.join(DATA_ROOT, "cic_vectorized.csv")

def get_pretrained_dirs():
    path, dirs, _ = next(os.walk(REPO_ROOT/"models"))
    return {
        d: os.path.join(path, d) for d in dirs
    }

