from config import prepare_config
from data_generation import data_generation
from data_preprocessing.main import data_preprocessing
from simulation.main import run_simulations
from testing import testing
from training import training
from validation import validation

config_settings = prepare_config()

#data_generation(config_settings)

#data_preprocessing(config_settings)

#config_settings = validation(config_settings)

#training(config_settings)

testing(config_settings)

#run_simulations(config_settings)