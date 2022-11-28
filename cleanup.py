import os

from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.storage.blob import BlobServiceClient
from azure.ai.ml import MLClient

def delete_resource_group():

    # Instantiating env variables and params for resource group
    subscription_id = os.environ["SUBSCRIPTION_ID"]
    rg_name = os.environ["RESOURCE_GROUP_NAME"]
    
    try:
        # Creating client
        credentials = DefaultAzureCredential()
        client = ResourceManagementClient(credentials, subscription_id)

        # Deleting resource group
        print("Deleting Resource Group")
        delete_async_operation = client.resource_groups.begin_delete(rg_name)
        delete_async_operation.wait()
        print("\nDeleted: {}".format(rg_name))
    
    except Exception as error:
        print(error)


def delete_workspace():

    # Instantiating env variables
    subscription_id = os.environ["SUBSCRIPTION_ID"]
    resource_group = os.environ["RESOURCE_GROUP_NAME"]
    workspace_name = os.environ["WORKSPACE_NAME"]

    try:
        # Creating client
        credentials = DefaultAzureCredential()
        client = MLClient(credentials, subscription_id, resource_group)

        # Deleting workspace
        print("Deleting Workspace")
        delete_async_operation = client.workspaces.begin_delete(workspace_name, delete_dependent_resources=True)
        delete_async_operation.wait()
        print("\nDeleted: {}".format(workspace_name))
    
    except Exception as error:
        print(error)


def delete_container():

    # Instantiating env variables
    storage_account = os.environ["STORAGE_ACCOUNT_NAME"]
    container_name = os.environ["CONTAINER_NAME"]
    account_url = "https://" + storage_account + ".blob.core.windows.net"

    try:
        # Creating client
        credentials = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url, credential=credentials)
        container_client = blob_service_client.get_container_client(container=container_name)

        # Delete the blob container
        print("Deleting blob container...")
        container_client.delete_container()

    except Exception as error:
        print(error)


def main():

    delete_container()
    delete_workspace()
    delete_resource_group()


if __name__ == '__main__':

    main()