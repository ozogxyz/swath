# Swath for Earth Observation ML

Rationale: 
- We need a quick way to load ONNX models and make inference on edge devices. These devices often require low power. Solafune's precipitation competition penalizes the expense of using big models, for example. 

- Managing virtual environments, geopandas and python etc is a nightmare on edge devices. Idea is to have a self contained executable that will leverage GDAL and load the rasters, make the transformations, load the ONNX model and run the inference. 

Roadmap: 
- Ideally this project would be written in Jai as a self contained program from compilation to configuration. Generate bindings for GDAL, ONNX and make an end-to-end one unit of EO-ML.
- Add NVIDIA Jetson CUDA support and/or bindings for fast inference on edge devices.

