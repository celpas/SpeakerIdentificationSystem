# ****************************
# Weights
WEIGHTS_RECOGNIZER      = 'data/weights/checkpoint_100.pth'
#WEIGHTS_DEEPXI         = 'data/weights/resnet-1.1n'
WEIGHTS_DEEPXI          = 'data/weights/mhanet-1.1c'
WEIGHTS_DENOISER        = 'data/weights/master64.th'

# ****************************
# Use speech enhancement network
ENHANCE     = False
ENHANCE_DRY = 0.1

# ****************************
# Data
SAMPLES_DIR = 'samples'
CONVERTER   = 'sox'  # or sox

# ****************************
# Preprocessing
USE_LOGSCALE = True
USE_NORM     = False
USE_DELTA    = False
USE_SCALE    = False
SAMPLE_RATE  = 16000
FILTER_BANK  = 40
WINLEN       = 0.025
WINSTEP      = 0.01