# BiRefNet: Project Improvements To-Do List

## Code Quality
- [ ] Add type hints throughout the codebase for better IDE support
- [ ] Refactor config.py to use dataclasses or OmegaConf
- [ ] Implement memory profiling for large datasets
- [ ] Improve error handling and logging
- [ ] Add mixed precision training by default

## Features
- [ ] Add export support for more deployment formats (TensorRT)
- [ ] Implement real-time visualization tools during training
- [ ] Add model pruning and quantization for edge deployment
- [ ] Support training resumption with automatic checkpointing
- [ ] Implement gradient accumulation for limited memory scenarios
- [x] Add support for multiple output segmentation maps
  - Set `config.num_output_channels = N` to output N segmentation maps
  - Loss functions updated to support multi-channel predictions
  - Supports both single-channel GT and multi-channel GT

## Documentation & Testing
- [ ] Add comprehensive docstrings to all functions and classes
- [ ] Create a complete API reference
- [ ] Create architecture diagrams for better understanding
- [ ] Implement unit tests for core functions
- [ ] Add validation image visualization during training
- [ ] Implement benchmark tests across different hardware

## Deployment
- [ ] Create Docker containers for easy deployment
- [ ] Implement a RESTful API server for predictions
- [ ] Add TorchServe integration for model serving
- [ ] Create mobile-optimized model variants
- [ ] Implement batch processing for large datasets

## Modern Best Practices
- [ ] Implement DistributedDataParallel for more efficient multi-GPU training
- [x] Add experiment tracking (MLflow or Weights & Biases)
- [ ] Implement CI/CD pipeline for testing
- [ ] Use reproducible environment setup (conda-lock or poetry)
- [ ] Add model versioning and experiment tracking