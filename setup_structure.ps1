# Folders to create
$folders = @(
    "data\raw",
    "data\splits",
    "preprocessing",
    "models\saved_models",
    "federated",
    "evaluation",
    "webapp\templates",
    "webapp\static",
    "notebooks",
    "utils"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
}

# Add __init__.py to package folders (skip data, notebooks, webapp/templates, webapp/static)
$package_folders = @(
    "preprocessing",
    "models",
    "models\saved_models",
    "federated",
    "evaluation",
    "webapp",
    "utils"
)

foreach ($folder in $package_folders) {
    New-Item -ItemType File -Path "$folder\__init__.py" -Force | Out-Null
}

# Starter README.md
@"
# Federated Deep Learning for Diabetic Retinopathy Detection

Final year engineering major project implementing federated learning
for diabetic retinopathy detection using CoAtNet-1, with differential
privacy (Opacus) and support for FedAvg/FedProx strategies.

## Structure
- data/ - dataset splits (raw datasets live on Google Drive)
- preprocessing/ - image processing, augmentation, SMOTE
- models/ - model definitions and saved checkpoints
- federated/ - client, server, FedAvg, FedProx implementations
- evaluation/ - metrics, Grad-CAM, visualizations
- webapp/ - Flask-based demo application
- notebooks/ - experiment notebooks
- utils/ - helper functions, experiment tracking
"@ | Out-File -FilePath "README.md" -Encoding utf8

# Starter requirements.txt (auto-generated from current environment)
pip freeze > requirements.txt

Write-Host "Folder structure created successfully." -ForegroundColor Green