# Rust-Neural-Network-Examples
this project is made of three different components:
1. gpu-accel(lib): GPU acceleration using WGPU/Tokio
2. nn-backbone(lib): Neural network implementation
3. examples(main): examples using 1 and 2

# Current status:
- [x] Basic Neural Network
- [x] MNIST implementation
- [ ] Backpropagation implementation
- [ ] Learning implementation
- [ ] CNN implementation
- [ ] RNN implementation

# Known Issue:
- Every test was done on **AMD GPU** (RX 7900 XTX). and it seems to have 12 GPU compute iteration limit(Command Buffer Queue). if test iteration goes over 12, it will likely hang.