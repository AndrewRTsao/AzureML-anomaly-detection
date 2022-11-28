# Initial project related variables
export LOCAL_PROJECT="" # Fully qualified path of your locally cloned project

# Azure service principal
export AZURE_TENANT_ID="" # Tenant Id that can be found in Azure Portal > Azure Active Directory > Properties
export AZURE_CLIENT_ID="" # Client ID (appID when you generated your Azure service principal)
export AZURE_CLIENT_SECRET="" # Client secret (password when you generated your Azure service principal)

# Azure account information
export SUBSCRIPTION_ID="" # Subscription ID that can be found under your Azure Portal > Subscriptions
export AZURE_LOCATION="" # Location where you'd like to create your Resource Group and Workplace (e.g. "westus" or "eastus"). See: https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/?products=machine-learning-service 
export RESOURCE_GROUP_NAME="" # Name that you'd like to give your Resource Group
export STORAGE_ACCOUNT_NAME="" # Name of your storage account (NOTE: must be unique across Azure!)
export WORKSPACE_NAME="" # Name that you'd like to give your Workspace
export CPU_CLUSTER="" # Name that you'd like to give your CPU compute cluster resource
export CONTAINER_NAME="" # Name of the ABS bucket or container, in lowercase, where you would like to upload / store your assets (e.g. training data, model, etc.)

# Input dataset information
export TRAIN_NAME="Train.csv" # Rename this if you end up changing the name of the Train.csv file (from unzipping the Kaggle dataset)
