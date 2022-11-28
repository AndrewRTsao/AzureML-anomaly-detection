import os
import subprocess
import logging
from tenacity import retry, wait_fixed, stop_after_attempt
from tenacity import RetryError

from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.storage.blob import BlobServiceClient
from azure.ai.ml.entities import Workspace, AmlCompute
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component


def log_attempt_number(retry_state):

    # Capture the result of the last retry attempt (where retrying has been implemented)
    print(f"Retrying: {retry_state.attempt_number}...")
    logging.error(f"Retrying: {retry_state.attempt_number}...")


def create_resource_group():

    # Instantiate env variables and params for resource group
    subscription_id = os.environ["SUBSCRIPTION_ID"]
    region = os.environ["AZURE_LOCATION"]
    rg_name = os.environ["RESOURCE_GROUP_NAME"]
    rg_params = {'location': region}

    try:
        # Create client
        credentials = DefaultAzureCredential()
        client = ResourceManagementClient(credentials, subscription_id)

        # Create resource group
        print("Creating Resource Group")
        client.resource_groups.create_or_update(rg_name, rg_params)
        print("\nResource group created: {}".format(rg_name))

    except Exception as error:
        print(error)


def create_storage_account():

    # Instantiate env variables
    subscription_id = os.environ["SUBSCRIPTION_ID"]
    resource_group = os.environ["RESOURCE_GROUP_NAME"]
    storage_account_name = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = os.environ["CONTAINER_NAME"]
    region = os.environ["AZURE_LOCATION"]

    try:
        # Create client
        credentials = DefaultAzureCredential()
        client = StorageManagementClient(credentials, subscription_id)

        availability_result = client.storage_accounts.check_name_availability(
            { "name": storage_account_name }
        )

        # Check if storage account name is available (needs to be unique across Azure)
        if not availability_result.name_available:
            print(f"Storage name {storage_account_name} is already in use. Terminating run and try again with another name.")
            os._exit(0)

        # Provision account if storage account name is available
        poller = client.storage_accounts.begin_create(resource_group, storage_account_name,
            {
                "location" : region,
                "kind": "StorageV2",
                "sku": {"name": "Standard_LRS"}
            }
        )

        # Wait for long running operation to complete
        account_result = poller.result()
        print(f"Provisioned storage account {account_result.name}")

        # Retrieve account's primary access key and generate a connection string
        keys = client.storage_accounts.list_keys(resource_group, storage_account_name)
        print(f"Primary key for storage account: {keys.keys[0].value}")
        conn_string = f"DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName={storage_account_name};AccountKey={keys.keys[0].value}"
        print(f"Connection string: {conn_string}")

        # Provision the blob container in the account and ensure that it's publicly accessible to be read
        container = client.blob_containers.create(resource_group, storage_account_name, container_name, {"public_access": 'Container'})
        print(f"Provisioned blob container {container.name}")

    except Exception as error:
        print(error)


def create_workspace():

    # Instantiate env variables
    subscription_id = os.environ["SUBSCRIPTION_ID"]
    resource_group = os.environ["RESOURCE_GROUP_NAME"]
    workspace_name = os.environ["WORKSPACE_NAME"]
    region = os.environ["AZURE_LOCATION"]
    cpu_compute_target = os.environ["CPU_CLUSTER"]

    try:
        # Create client
        credentials = DefaultAzureCredential()
        client = MLClient(credentials, subscription_id, resource_group)

        # Establish workspace parameters and create workspace
        ws = Workspace(
            name=workspace_name,
            location=region,
            display_name=workspace_name,
            description="Workspace created for MSFT-anomaly-python delivery example",
            hbi_workspace=False,
            tags=dict(purpose="Continual delivery example"),
        )

        print("Creating Workspace")
        workspace_poller = client.workspaces.begin_create(ws)
        workspace_result = workspace_poller.result()
        print(workspace_result)
        print("\nWorkspace created: {}".format(workspace_name))

        client = MLClient(credentials, subscription_id, resource_group, workspace_name)
        print("Updated client with new default workspace")

        # Create cpu compute cluster
        try:
            client.compute.get(cpu_compute_target)
            print("Compute cluster aleady exists!")
        
        except Exception:
            print("Creating a new CPU compute target...")
            compute = AmlCompute(
                name=cpu_compute_target,
                type="amlcompute",
                size="STANDARD_D2_V2",
                location=region,
                min_instances=0, 
                max_instances=4,
                idle_time_before_scale_down=120,
            )

            poller = client.compute.begin_create_or_update(compute)
            compute_resource = poller.result()
            print(f"Provisioned compute cluster {compute_resource.name}")

    except Exception as error:
        print(error)


@retry(wait=wait_fixed(60), stop=stop_after_attempt(5), after=log_attempt_number)
def upload_data():

    # Instantiate env variables
    local_project = os.environ["LOCAL_PROJECT"]
    data_path = local_project + "/data"
    input_file = "Train.csv"
    upload_file_path = os.path.join(data_path, input_file)
    storage_account = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = os.environ["CONTAINER_NAME"]
    account_url = "https://" + storage_account + ".blob.core.windows.net"

    try:
        print("Granting Storage Blob Data Contributor to service account so it can access / upload blob to container")
        permissions_script = os.path.join(local_project, "update_permissions.sh")
        subprocess.call(permissions_script)
        print("Storage Blob Data Contributor role granted")

        # Creating BlobServiceClient and then blob client
        print("Retrieving Azure credentials")
        default_credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url, credential=default_credential)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=input_file)
        print("\nUploading to Azure Storage as blob:\n\t" + input_file)

        # Upload file
        with open(file=upload_file_path, mode="rb") as data:
            blob_client.upload_blob(data)

    except Exception as error:
        print(error)


@retry(wait=wait_fixed(60), stop=stop_after_attempt(5), after=log_attempt_number)
def run_pipeline():
    
    # Instantiate env variables
    local_project = os.environ["LOCAL_PROJECT"]
    subscription_id = os.environ["SUBSCRIPTION_ID"]
    resource_group = os.environ["RESOURCE_GROUP_NAME"]
    storage_account = os.environ["STORAGE_ACCOUNT_NAME"]
    workspace = os.environ["WORKSPACE_NAME"]
    container_name =  os.environ["CONTAINER_NAME"]
    train_file = os.environ["TRAIN_NAME"]
    cpu_compute_target = os.environ["CPU_CLUSTER"]

    account_url = "https://" + storage_account + ".blob.core.windows.net"
    container_path = account_url + "/" + container_name
    container_file_path = container_path + "/" + train_file

    # Load pipeline components
    print("Loading components")
    prepare_data_component = load_component(source=local_project + "/src/prep/prep.yaml")
    train_model_component = load_component(source=local_project + "/src/train/train.yaml")
    predict_outliers_component = load_component(source=local_project + "/src/predict/predict.yaml")


    # Defining pipeline with several components - ingesting / prepping data, training the anomaly detection model, and predicting outliers.
    @pipeline(
        default_compute=cpu_compute_target,
    )
    def anomaly_detection_pipeline(pipeline_input_data):
        prepare_data = prepare_data_component(
            raw_data=pipeline_input_data
        )
        train_model = train_model_component(
            training_data=prepare_data.outputs.prep_data
        )
        predict_outliers = predict_outliers_component(
            input_data=prepare_data.outputs.prep_data,
            input_model=train_model.outputs.model_output
        )

        return {
            "pipeline_job_prepped_data": prepare_data.outputs.prep_data,
            "pipeline_job_trained_model": train_model.outputs.model_output,
            "pipeline_job_outlier_predictions": predict_outliers.outputs.output_result,
        }

    
    def compile_and_trigger_pipeline():

        # Defining the ABS container we created earlier as the input datasource for our Azure ML pipeline
        # Need to use URI_FILE instead of URI_FOLDER with blob path because of the following bug: https://github.com/Azure/azure-sdk-for-python/issues/27318
        print("Specifying input datasource")

        input_ds = Input(
            type = AssetTypes.URI_FILE,
            path = container_file_path
        )

        # Create the pipeline job definition and update output settings
        print("Building the pipeline")
        pipeline_job = anomaly_detection_pipeline(pipeline_input_data=input_ds)

        # Submit job to workspace
        print("Kicking off pipeline")
        pipeline_job = ml_client.jobs.create_or_update(
            pipeline_job, experiment_name="Anomaly detection pipeline"
        )
        pipeline_job
        print("Waiting for pipeline job to complete")
        ml_client.jobs.stream(pipeline_job.name)
        print("Pipeline job complete")


    # Creating client needed to submit Azure ML jobs
    credentials = DefaultAzureCredential()
    print("Instantiate MLClient with new workspace")
    ml_client = MLClient(credentials, subscription_id, resource_group, workspace)

    # Run the pipeline
    compile_and_trigger_pipeline()


def main():

    # Initial setup of Azure environment
    create_resource_group()
    create_storage_account()
    create_workspace()

    # Attempt to upload data up to 5 times and wait 60 seconds between attempts until resources from initial setup become available
    print("Attempting to upload local data to Azure Blob")
    try: 
        upload_data()

    except (Exception, RetryError) as error:
        print("Unable to upload data despite multiple reattempts")
        print(error)

    print("Data has been successfully uploaded to Azure")

    # Attempt to run pipeline up to 5 times and wait 60 seconds between attempts until resources are available from initial setup
    # (e.g. initial credentials to ACR might fail, data access might not be fully available, etc.)
    print("Attempting to compile and trigger initial Azure ML pipeline")
    try:
        run_pipeline()

    except (Exception, RetryError) as error:
        print("Pipeline failed 5 times - ending now")
        print(error)


if __name__ == '__main__':

    main()
