# Getting Started

1. Download the datasets from the BigMart Sales Data [Kaggle project](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data) either directly or by using the Kaggle CLI. Then, unzip the archive and copy the resulting CSV files into the *./data* directory of this project. 

```sh 
kaggle datasets download -d brijbhushannanda1979/bigmart-sales-data
```

2. Create a virtual environment and pip install requirements.txt locally.

```sh 
pip install --trusted-host pypip.python.org -r requirements.txt
```

3. Create an Azure service principal either through [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/create-an-azure-service-principal-azure-cli), [PowerShell](https://learn.microsoft.com/en-us/azure/active-directory/develop/howto-authenticate-service-principal-powershell) or [the portal](https://learn.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal). This service principal *will need to have sufficient privileges* (e.g. Owner) on your subscription.

```sh 
az ad sp create-for-rbac -n test-account --role Owner --scopes /subscriptions/<subscription_ID>
```

*NOTE: Make a note of the output here, which you will need for the next step (appID = client_ID, password = client_secret).*

4. Fill out the environment variables in **env.sh** and source the file.

```sh 
source env.sh
```

4b. (Optional) Update the `outlier_columns` in **/src/prep/prep_component.py** if you wish to predict outliers across different columns / dimensions than what has been pre-selected (e.g. Item_MRP / Item_Outlet_Sales). At least *two or more* columns must be selected in `outlier_columns` for the model to work. 

4c. (Optional) Update the `model_type` and/or `hyperparameters` to pass to your pyod model under **/src/train/train_component.py** if you wish to change the specific algorithm or associated hyperparameters being used when training the anomaly detection model (e.g. ABOD, contamination=0.05). Otherwise, leave the defaults. For more details about possible algorithm types, please refer to the [pyod documentation](https://pyod.readthedocs.io/en/latest/pyod.models.html).

5. Run the **run_pipeline.py** script to trigger the end-to-end Azure ML pipeline (will setup an environment / workspace from scratch and build your pipeline).

```sh
python run_pipeline.py
```

*NOTE: Pipeline will take approximately 30 minutes to one hour to complete*

6. (Optional) If you would like, run the **cleanup.py** script once you're done and/or if you don't need the underlying pipeline assets / resources anymore.

```sh
python cleanup.py
```
