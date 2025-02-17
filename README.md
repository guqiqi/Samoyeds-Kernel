# Samoyeds-Kernel Startup
``Samoyeds-Kernel`` is a library for dual-side sparse matrix multiplication

## Install

### Pre-requisites
Samoyeds-Kernel requires the following dependencies:
- CUDA 11.4+
- CMake 3.18+

### Get the Code

```shell
git clone --recurse-submodules <TBD> Samoyeds-Kernel
cd Samoyeds-Kernel
```

### Build
To build the Samoyeds-Kernel, you need to run the following script.
```shell
./build.sh
```

> If you meet a problem like this:
> ```shell
> Policy "CMP0104" is not known to this version of CMake
> ```
> Please, comment this line `cmake_policy(SET CMP0104 OLD)` in `./benchmark/third_party/venom/include/sputnik/CMakeLists.txt`

### Run
To build the Samoyeds-Kernel, you need to run the following script.
```shell
./run.sh
```

