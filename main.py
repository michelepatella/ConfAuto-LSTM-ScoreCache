from config import prepare_config
from data_generation import data_generation
from data_preprocessing.main import data_preprocessing
from simulation.main import run_simulations
from testing import testing
from training import training
from validation import validation


# PIPELINE
# 0. PREPARE CONFIGURATION SETTINGS
config_settings = prepare_config()

# 1. GENERATE SYNTHETIC DATA
data_generation(config_settings)

# 2. PREPROCESS GENERATED DATA
data_preprocessing(config_settings)

# 3. FIND THE BEST HYPERPARAMETERS
config_settings = validation(config_settings)

# 4. TRAIN THE MODEL
training(config_settings)

# 5. TEST THE STANDALONE MODEL
testing(config_settings)

# 6. COMPARE THE FRAMEWORK AGAINST BASELINE CACHES
run_simulations(config_settings)