import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load

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


def parse_args():

    # setting up argparse
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--input_data", type=str, help="Path containing data for scoring"
    )
    parser.add_argument(
        "--input_model", type=str, help="Input path for saved model"
    )
    parser.add_argument(
        "--output_result", type=str, default="./", help="Output path for outlier prediction results"
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


def predict_outliers(input_data, input_model, output_result):

    # Reading previously prepared / cleaned data to perform outlier predictions
    data_file = get_file(input_data)
    data = pd.read_csv(data_file)

    print("=====Check data=====")
    print(data)
    print(data.notna().count())
    
    # Retrieving previously trained pyod model
    algo = load((Path(input_model) / "clf.joblib"))

    # Converting df to numpy array to avoid any weird downstream numba issues with certain modules
    data_columns = list(data.columns) # Storing column names for later use when converting from numpy array back to df
    data = data.to_numpy()

    # Ensuring that np has float dtype and not object dtype to again avoid downstream issues with numba
    data = np.vstack(data[:, :]).astype(np.float)

    print("Predicting outliers and outlier scores from data")
    y_pred = algo.predict(data)
    scores_pred = algo.decision_function(data) * -1
    print("Finished predictions and calculating prediction scores")

    # Converting numpy arrays to df
    data = pd.DataFrame(data, columns=data_columns)
    outlier_df = pd.DataFrame(y_pred, columns=['outlier'])
    outlier_score_df = pd.DataFrame(scores_pred, columns=['outlier_score'])

    # Concatenate outlier predictions and prediction scores to original dataset, then write the resulting df to the output
    data = pd.concat([data, outlier_df, outlier_score_df], axis=1)

    print("writing final output")
    data.to_csv((Path(output_result) / "data_with_outlier_predictions.csv"))


def main(args):

    predict_outliers(args.input_data, args.input_model, args.output_result)


if __name__ == "__main__":

    # parse args and pass it to the main function
    args = parse_args()
    main(args)