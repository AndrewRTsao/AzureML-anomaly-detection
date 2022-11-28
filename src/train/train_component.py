import argparse
import pandas as pd
from pathlib import Path
from joblib import dump

from pyod.models.abod import ABOD
from pyod.models.alad import ALAD
from pyod.models.anogan import AnoGAN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.auto_encoder_torch import AutoEncoder as AutoEncoderTorch
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.cd import CD
from pyod.models.copod import COPOD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ecod import ECOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.lunar import LUNAR
from pyod.models.lscp import LSCP
from pyod.models.mad import MAD
from pyod.models.mcd import MCD
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.rgraph import RGraph
from pyod.models.rod import ROD
from pyod.models.sampling import Sampling
from pyod.models.sod import SOD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.sos import SOS
from pyod.models.suod import SUOD
from pyod.models.vae import VAE

# Specify the model type (from list of available pyod algorithms) and corresponding hyperparameters that you would like to use for the model
model_type = "ABOD" # Should be a string
hyperparameters = {"contamination": 0.05, "n_neighbors": 5} # Should be a dict


def parse_args():

    # setting up argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--training_data", type=str, help="Path to prepared training data"
    )
    parser.add_argument(
        "--model_output", type=str, help="Path of saved model"
    )

    # parse and return args
    args = parser.parse_args()
    return args


def get_file(f):

    f = Path(f)
    if f.is_file():
        return f
    else:
        files = list(f.iterdir())
        if len(files) == 1:
            return files[0]
        else:
            raise Exception("********This path contains more than one file*******")


def get_algo(algorithm, params):

    algo_object = None
    algo_params = params

    if algorithm == "ABOD":
        algo_object = ABOD(**algo_params)
    elif algorithm == "ALAD":
        algo_object = ALAD(**algo_params)
    elif algorithm == "AnoGAN":
        algo_object = AnoGAN(**algo_params)
    elif algorithm == "AutoEncoder":
        algo_object = AutoEncoder(**algo_params)
    elif algorithm == "AutoEncoderTorch":
        algo_object = AutoEncoderTorch(**algo_params)
    elif algorithm == "CBLOF":
        algo_object = CBLOF(**algo_params)
    elif algorithm == "COF":
        algo_object = COF(**algo_params)
    elif algorithm == "CD":
        algo_object = CD(**algo_params)
    elif algorithm == "COPOD":
        algo_object = COPOD(**algo_params)
    elif algorithm == "DeepSVDD":
        algo_object = DeepSVDD(**algo_params)
    elif algorithm == "ECOD":
        algo_object = ECOD(**algo_params)
    elif algorithm == "FeatureBagging":
        algo_object = FeatureBagging(**algo_params)
    elif algorithm == "GMM":
        algo_object = GMM(**algo_params)
    elif algorithm == "HBOS":
        algo_object = HBOS(**algo_params)
    elif algorithm == "IForest":
        algo_object = IForest(**algo_params)
    elif algorithm == "INNE":
        algo_object = INNE(**algo_params)
    elif algorithm == "KDE":
        algo_object = KDE(**algo_params)
    elif algorithm == "KNN":
        algo_object = KNN(**algo_params)
    elif algorithm == "AverageKNN":
        algo_object = KNN(method='mean', **algo_params)
    elif algorithm == "LMDD":
        algo_object = LMDD(**algo_params)
    elif algorithm == "LOCI":
        algo_object = LOCI(**algo_params)
    elif algorithm == "LODA":
        algo_object = LODA(**algo_params)
    elif algorithm == "LOF":
        algo_object = LOF(**algo_params)
    elif algorithm == "LUNAR":
        # Check if contamination is in algo_params because it's not a valid key for this algorithm. If so, pop it. 
        print("popping contamination for LUNAR")
        algo_params.pop("contamination")
        print("new kwargs for LUNAR")
        print(algo_params)
        
        # Pass in updated kwargs
        algo_object = LUNAR(**algo_params)
    elif algorithm == "LSCP":       
        # Check if detector_list is being passed in as a hyperparameter (incomplete - need to parse and create estimators still). 
        # If not, initialize a set of detectors for LSCP as it's required. 
        if 'detector_list' not in algo_params:
            print("no detector_list found in algo_params so using predefined set")
            detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15), 
            LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30), LOF(n_neighbors=35), 
            LOF(n_neighbors=40), LOF(n_neighbors=45), LOF(n_neighbors=50)]

            print("Printing detector_list")
            print(detector_list)

            # Pass in correct params
            algo_object = LSCP(detector_list, **algo_params)
        else:
            print("detector_list found in algo_params")
            print("Printing detector_list")
            print(algo_params["detector_list"])

            # Pass in correct params
            algo_object = LSCP(**algo_params)
    elif algorithm == "MAD":
        # Check if contamination is in algo_params because it's not a valid key for this algorithm. If so, pop it. 
        print("popping contamination for MAD")
        algo_params.pop("contamination")
        print("new kwargs for MAD")
        print(algo_params)

        # Pass in updated kwargs
        algo_object = MAD(**algo_params)
    elif algorithm == "MCD":
        algo_object = MCD(**algo_params)
    elif algorithm == "MO_GAAL":
        algo_object = MO_GAAL(**algo_params)
    elif algorithm == "OCSVM":
        algo_object = OCSVM(**algo_params)
    elif algorithm == "PCA":
        algo_object = PCA(**algo_params)
    elif algorithm == "RGraph":
        algo_object = RGraph(**algo_params)
    elif algorithm == "ROD":
        algo_object = ROD(**algo_params)
    elif algorithm == "Sampling":
        algo_object = Sampling(**algo_params)
    elif algorithm == "SOD":
        algo_object = SOD(**algo_params)
    elif algorithm == "SO_GAAL":
        algo_object = SO_GAAL(**algo_params)
    elif algorithm == "SOS":
        algo_object = SOS(**algo_params)
    elif algorithm == "SUOD":
        # Check if base_estimators is being passed in as a hyperparameter.
        # If not, initialize a group of base estimators / outlier detectors for SUOD as it's required. 
        if 'base_estimators' not in algo_params:
            print("no base_estimators found in algo_params")
            base_estimators = [LOF(n_neighbors=15), LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=35), 
            COPOD(), IForest(n_estimators=100), IForest(n_estimators=200)]

            print("Printing base_estimators")
            print(base_estimators)

            # Pass in correct params
            algo_object = SUOD(base_estimators, **algo_params)   
        else:
            print("base_estimators found in algo_params")
            print("Printing base_estimators")
            print(algo_params["base_estimators"])

            # Pass in correct params
            algo_object = SUOD(**algo_params)
    elif algorithm == "VAE":
        algo_object = VAE(**algo_params)
    else:
        pass
        
    return algo_object


def train_model(training_data, model_output):

    # Reading prepared / cleaned training data
    data_file = get_file(training_data)
    data = pd.read_csv(data_file)

    # Retrieving model based on algorithm type and fit the model
    print("Retrieving model")
    print(model_type)
    print(hyperparameters)

    algo = get_algo(model_type, hyperparameters)

    print("Printing algo object")
    print(algo)

    print("=====Check data=====")
    print(data)
    print(data.notna().count())

    print("Fitting model")
    algo.fit(data)
    print("Model has been fit")

    # Write model output

    dump(algo, (Path(model_output) / "clf.joblib"))


def main(args):
    
    train_model(args.training_data, args.model_output)


if __name__ == "__main__":

    # parse args and pass it to the main function
    args = parse_args()
    main(args)

