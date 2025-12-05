## Membership Inference Attacks

This project aims to evaluate MIAs through [Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter) tool.

The experiment is divided into two files.
1. setup.py: provide all the necessary configurations. 
2. membership-inference_locations.ipynb: provide all the analysis for MIA.


Environment steup with UV
```
uv venv --python 3.12 .venv
```

### Configs
First, fork the Privacy Meter from GitHub page to your own account. This ensures you can modify and customize the code as needed. Then, run the setup.py script that clone your forked repo, clean and install all required dependencias and create the config file for the data. Change configs according the data and model you want to test. For this experiment, tabular Locations (Bangkok) dataset is used.

Before runing the script, make sure to update the github_username field in the main function with your own GitHub username..
```
github_user = "YOUR_USERNAME"
```

The script automatically removes all CUDA-related dependencies from requirements.txt.
If your machine does support CUDA, simply comment out or disable the remove_cuda_packages() function call to keep GPU support enabled.

```
uv run setup.py
```


### MIA exepriment
After completing all configurations, execute the notebook membership-inference_locations.ipynb cells to run the experiments. Review and analyze the output of each cell to understand the results.
