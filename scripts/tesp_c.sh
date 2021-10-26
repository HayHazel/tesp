#!/bin/bash

if [[ -z ${INSTDIR} ]]; then
  . environment
fi

echo
echo ++++++++++++++  Compiles all TESP software and libraries  ++++++++++++++
echo
# Install all pip libraries
#pip3 install wheel colorama glm seaborn matplotlib networkx==2.3 numpy pandas pulp pyutilib==5.8.0 pyomo==5.6.8 PYPOWER scikit-learn scipy tables h5py xlrd
pip3 install wheel colorama glm seaborn matplotlib networkx numpy pandas pulp pyutilib==5.8.0 pyomo==5.6.8 PYPOWER scikit-learn scipy tables h5py xlrd

#develop tesp api
cd "${TESPDIR}/src/tesp_support" || exit
pip3 install -e .
#develop psst api
cd "${REPODIR}/psst" || exit
pip3 install -e .

#  pip3 install tesp_support --upgrade
#  pip3 install psst --upgrade

cd "${TESPDIR}/scripts/build" || exit
if [[ $1 == "develop" ]]; then

  echo Building and Installing FNCS
  ./fncs_b.sh clean > fncs.log
  echo Building and Installing FNCS for Java
  ./fncs_java_b.sh clean > fava_java.log
  echo Building and Installing HELICS
  ./helics2_b.sh clean > helics2.log
  echo Building and Installing KLU
  ./klu_b.sh clean > klu.log
  echo Building and Installing Gridlabd
  ./gridlabd_b.sh clean > gridlabd.log
  echo Building and Installing EngeryPlus
  ./energyplus_b.sh clean > energyplus.log
  echo Building and Installing EnergryPlus for Java
  ./energyplusj_b.sh clean > energyplusj.log
  echo Building and Installing NS-3
  ./ns-3_b.sh clean > ns3.log
  echo Building and Installing Ipopt with ASL and MUMP
  ./ipopt_b.sh clean > ipopt.log
else

  wget https://mepas.pnnl.gov/FramesV1/Install/tesp_binaries.zip
  cd "${INSTDIR}" exit
  unzip tesp_binaries.zip .
fi

# creates the necessary links and cache to the most recent shared libraries found
# in the directories specified on the command line, in the file /etc/ld.so.conf,
# and in the trusted directories (/lib and /usr/lib).
sudo ldconfig

echo
echo ++++++++++++++  Build and compile TESP is complete!  ++++++++++++++
echo
./versions.sh