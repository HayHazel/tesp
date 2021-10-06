#!/bin/bash


#sudo apt-get -y install openssh-server
#sudo nano /etc/ssh/sshd_config
#Once you open the file, find and change the uncomment line: # Port 22 

# build tools
sudo apt-get -y install apt-utils
sudo apt-get -y install git
sudo apt-get -y install build-essential
sudo apt-get -y install autoconf
sudo apt-get -y install libtool
sudo apt-get -y install libjsoncpp-dev
sudo apt-get -y install gfortran
sudo apt-get -y install cmake
sudo apt-get -y install subversion
sudo apt-get -y install wget
sudo apt-get -y install pkg-config
sudo apt-get -y install unzip

# Java support
sudo apt-get -y install openjdk-11-jre-headless
sudo apt-get -y install openjdk-11-jdk-headless
sudo ln -s /usr/lib/jvm/java-11-openjdk-amd64 /usr/lib/jvm/default-java

# for HELICS and FNCS
sudo apt-get -y install libzmq5-dev
sudo apt-get -y install libczmq-dev
sudo apt-get -y install libboost-dev

# for GridLAB-D
sudo apt-get -y install libxerces-c-dev
sudo apt-get -y install libhdf5-serial-dev
sudo apt-get -y install libsuitesparse-dev
# end users replace libsuitesparse-dev with libklu1, which is licensed LGPL

# for solvers Ipopt/cbc used by AMES/Agents
sudo apt-get -y install coinor-cbc
sudo apt-apt -y install coinor-libipopt-dev
sudo apt-get -y install liblapack-dev
sudo apt-get -y install libmetis-dev

# Python support
# if not using miniconda (avoid Python 3.7 on Ubuntu for now)
sudo apt-get -y install python3-pip
sudo apt-get -y install python3-tk

#              coinor-libcbc-dev \
#              gosu \
##              libboost-dev \
##              libboost-filesystem-dev \
##              libboost-program-options-dev \
##              libboost-signals-dev \
##              libboost-test-dev \
#              lsof \
#              make \
#              python-minimal \
#              python-pip \
#              python3 \
#              python3-dev \
##              swig \
#              uuid-dev \

# Set create directory structure for grid repository and installed software
mkdir grid
cd grid || exit
mkdir repository
mkdir installed
mkdir software
# Download all relevant repositories
cd repository || exit
# Set your name and email address
#git config --global user.name "your user name"
#git config --global user.email "your email"
#git config --global credential.helper store

# FNCS
#develop for dsot
git clone -b develop https://github.com/FNCS/fncs.git
#feature/opendss for tesp
#git clone -b feature/opendss https://github.com/FNCS/fncs.git

# HELICS
git clone -b helics2 https://github.com/GMLC-TDC/HELICS-src
#git clone -b main https://github.com/GMLC-TDC/HELICS-src

# GRIDLAB
#develop - dec21 commit number for dsot
#ENV GLD_VERSION=6c983d8daae8c6116f5fd4d4ccb7cfada5f8c9fc
git clone -b develop https://github.com/gridlab-d/gridlab-d.git

# ENERGYPLUS
git clone -b fncs_9.3.0 https://github.com/FNCS/EnergyPlus.git

# TESP
git clone -b evolve https://github.com/pnnl/tesp.git
# need for back port of DSOT
git clone -b master https://stash.pnnl.gov/scm/tesp/tesp-private.git

# NS3
git clone https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev || exit
git clone -b feature/13b https://github.com/GMLC-TDC/helics-ns3 contrib/helics
cd ..
git clone https://github.com/gjcarneiro/pybindgen.git

# PSST
git clone https://github.com/ames-market/psst.git

# KLU SOLVER
svn export https://github.com/gridlab-d/tools/branches/klu-build-update/solver_klu/source/KLU_DLL

# Install snap Pycharm IDE for python
# sudo snap install pycharm-community --classic

# to Run pycharm
# pycharm-community &> ~/charm.log&

# Compile all relevant executables
./tesp/tespCompile.sh