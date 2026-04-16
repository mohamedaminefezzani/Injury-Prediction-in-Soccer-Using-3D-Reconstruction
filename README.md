# Injury-Prediction-in-Soccer-Using-3D-Reconstruction

This is an ongoing R&D project for mitigating injury risk for soccer player by leveraging SAM 3D's body reconstruction.

# Requirements
- python 3.12
- runtime environment with nvidia GPU (preferably with at least 12GB vram)
- [a clone or fork of SAM 3d body](https://github.com/facebookresearch/sam-3d-body) (you will need the source files as they are)
- [a Hugging Face token with access to the model](https://huggingface.co/facebook/sam3) saved as an environment variable (HF_TOKEN)

# Installation guide (Linux)
- Give permission to install.sh
`sudo chmod +x install.sh`
- Run the script
- `./install.sh`

The script will install all requirements.

Once done, run the import, loading and subprocess cells from the SAM notebook.

**NOTE: THE SCRIPT DOES NOT INSTALL DEPENDENCIES IN A SEPARATE VIRTUAL ENVIRONMENT**
