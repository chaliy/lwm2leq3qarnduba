## Overview

Uses Azure Functions to deploy prediction function

## Deploy

Pre-requisites:
- Azure CLI
- Azure Functions Core Tools

```sh
az login
az config param-persist on # (Optional)
az group create --name mlprod --location northeurope
az storage account create --name onr55m2crkbyuag7 --sku Standard_LRS -g mlprod
az functionapp create \
    --name onr55m2crkbyuag7 \
    --consumption-plan-location northeurope \
    --runtime python --runtime-version 3.9 --functions-version 4 \
     --os-type linux \
     --storage-account onr55m2crkbyuag7 \
     -g mlprod
func azure functionapp publish onr55m2crkbyuag7 -b remote
func azure functionapp logstream onr55m2crkbyuag7 --browser
```

## Cleanup

```sh
az group delete --name mlprod
```