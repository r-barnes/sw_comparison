# AnyDSL
Meta project to quickly build dependencies

## Building on Linux

Note: You must build LLVM from source to get AVX vectorization

Install prerequisites:

    sudo apt install git-svn llvm-8-dev

Switch to the `cmake-based-setup` branch:

    git checkout cmake-based-setup

Prepare the build:

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -GNinja ..
    ninja

## Building

```bash
git clone https://github.com/AnyDSL/anydsl.git
cd anydsl
cp config.sh.template config.sh
./setup.sh
```
You may also want to fine-tune ```config.sh```.

See [Build Instructions](https://anydsl.github.io/Build-Instructions.html) for more information.
