#!/bin/bash

# Copyright (C) 2021-2023 Battelle Memorial Institute
# file: tesp_c.sh

if [[ -z ${INSTDIR} ]]; then
  . "${HOME}/tespEnv"
fi

echo
echo "++++++++++++++  Compiling and Installing TESP software is starting!  ++++++++++++++"
echo

echo "Installing Python Libraries..."
which python > "${TESPBUILD}/tesp_pypi.log" 2>&1
pip list >> "${TESPBUILD}/tesp_pypi.log" 2>&1
pip install sphinx-jsonschema sphinxcontrib-bibtex recommonmark xarray >> "${TESPBUILD}/tesp_pypi.log" 2>&1
pip install pygccxml pybindgen PYPOWER PuLP Pyomo PyYAML plotly future networkx pyutilib >> "${TESPBUILD}/tesp_pypi.log" 2>&1
#pip install NREL-PySAM~=4.1.0 PyGLM~=2.7.0 tables~=3.8.0 >> "${TESPBUILD}/tesp_pypi.log" 2>&1

echo "Installing Python TESP API..."
cd "${TESPDIR}/src/tesp_support" || exit
pip3 install -e . > "${TESPBUILD}/tesp_api.log" 2>&1

echo "Installing Python PSST..."
cd "${REPODIR}/AMES-V5.0/psst" || exit
pip3 install -e . > "${TESPBUILD}/AMES-V5.0.log" 2>&1

#  pip3 install tesp_support --upgrade
#  pip3 install psst --upgrade

cd "${TESPBUILD}" || exit
if [[ $1 == "develop" ]]; then

  echo "Compiling and Installing CZMQ..."
  ./czmq_b.sh clean > czmq.log 2>&1

  echo "Compiling and Installing FNCS..."
  ./fncs_b.sh clean > fncs.log 2>&1

  echo "Compiling and Installing FNCS for Java..."
  ./fncs_j_b.sh clean > fncs_j.log 2>&1

  echo "Compiling and Installing HELICS..."
  ./HELICS-src_b.sh clean > HELICS-src.log 2>&1

  echo "Compiling and Installing KLU..."
  wget --no-check-certificate https://raw.githubusercontent.com/gridlab-d/tools/klu-build-update/solver_klu/solver_klu_x64.dll  > KLU_DLL.log 2>&1
  mv solver_klu_x64.dll "${INSTDIR}/bin/solver_klu.dll" > KLU_DLL.log 2>&1

  echo "Compiling and Installing Gridlabd..."
  ./gridlab-d_b.sh clean > gridlab-d.log 2>&1

  echo "Compiling and Installing EnergyPlus..."
  ./EnergyPlus_b.sh clean > EnergyPlus.log 2>&1

  echo "Compiling and Installing EnergyPlus for Java..."
  ./EnergyPlus_j_b.sh clean > EnergyPlus_j.log 2>&1

  echo "Compiling and Installing NS-3..."
  ./ns-3-dev_b.sh clean > ns-3-dev.log 2>&1

  echo "Compiling and Installing Ipopt with ASL and Mumps..."
  ./ipopt_b.sh clean > ipopt.log 2>&1

  echo "Compiling and Installing TMY3toTMY2_ansi..."
  cd "${TESPDIR}/data/weather/TMY2EPW/source_code" || exit
  gcc TMY3toTMY2_ansi.c -o TMY3toTMY2_ansi
  mv TMY3toTMY2_ansi "${INSTDIR}/bin"
else

  ver=$(cat "${TESPBUILD}/version")
  echo "Installing HELICS, FNCS, GridLabD, EnergyPlus, NS3, and solver binaries..."
  cd "${INSTDIR}" || exit
#  wget --no-check-certificate https://github.com/pnnl/tesp/releases/download/${ver}/tesp_binaries.zip
#  unzip tesp_binaries.zip > "${TESPBUILD}/tesp_binaries.log" 2>&1
#  rm tesp_binaries.zip
fi

cd "${TESPBUILD}" || exit
echo "Installing HELICS Python bindings..."
./HELICS-py.sh clean > HELICS-py.log 2>&1

echo "Installing TESP documentation..."
./docs_b.sh clean > docs.log 2>&1

# Creates the necessary links and cache to the most recent shared libraries found
# in the directories specified on the command line, in the file /etc/ld.so.conf,
# and in the trusted directories (/lib and /usr/lib).
# sudo ldconfig
echo
echo "TESP installation logs are found in ${TESPBUILD}"
echo "++++++++++++++  Compiling and Installing TESP software is complete!  ++++++++++++++"

cd "${TESPBUILD}" || exit
./versions.sh

echo
echo "++++++++++++++  TESP has been installed! That's all folks!  ++++++++++++++"
echo