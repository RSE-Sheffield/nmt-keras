import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
# Used to suppress numpy FutureWarnings, turn off in developer mode

from deepquest.train import main as train
from deepquest.predict import main as predict
from deepquest.score import main as score
