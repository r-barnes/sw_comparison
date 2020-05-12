# AnyDSL
Meta project to quickly build dependencies

## Building on Linux

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
git clone git@github.com:AnyDSL/anydsl.git
cd anydsl
cp config.sh.template config.sh
./setup.sh
```
You may also want to fine-tune ```config.sh```.
In particular, if you don't have a GitHub account with a working [SSH key](https://help.github.com/articles/generating-ssh-keys), set ```: ${HTTPS:=true}```.
This will clone all repositories via https.

See [Build Instructions](https://anydsl.github.io/Build-Instructions.html) for more information.
