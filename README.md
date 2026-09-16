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
