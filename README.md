# Rust-Neural-Network-Examples
this project is made of three different components:
1. gpu-accel(lib): GPU acceleration using WGPU/Tokio
2. nn-backbone(lib): Neural network implementation
3. examples(main): examples using 1 and 2

- currently contribution is disabled. if anyone is interested to join the development, please wait until *learning implementation* is done by repo owner.
- project_root/data folder is excluded from the project for licensing. you will have to find data to run examples.

# Current status:
- [x] Basic Neural Network
- [x] MNIST implementation
- [ ] Backpropagation implementation
- [ ] Learning implementation
- [ ] CNN implementation
- [ ] RNN implementation

# Run examples:
- run examples using following command:
```
# mnist
cargo run --bin mnist
```

# Known Issue:
- Every test was done on **AMD GPU** (RX 7900 XTX). and it seems to have 12 GPU compute iteration limit(Command Buffer Queue). if test iteration goes over 12, it will likely hang.