#!/bin/bash
# Copyright (C) 2017-2023 Battelle Memorial Institute
# file: runSGIP1a.sh

declare -r SCHED_PATH=$TESPDIR/data/schedules
declare -r EPLUS_PATH=$TESPDIR/data/energyplus

(export FNCS_BROKER="tcp://*:5570" && exec fncs_broker 6 &> broker1a.log &)
# this is based on SGIP1b, writing to different metrics file and assuming the market clearing is disabled
(export FNCS_CONFIG_FILE=eplus.yaml && exec energyplus -w $EPLUS_PATH/USA_AZ_Tucson.Intl.AP.722740_TMY3.epw -d output -r $EPLUS_PATH/SchoolDualController_f.idf &> eplus1a.log &)
(export FNCS_CONFIG_FILE=eplus_agent.yaml && exec eplus_agent 2d 5m SchoolDualController eplus_SGIP1a_metrics.json &> eplus_agent1a.log &)
(export FNCS_FATAL=YES && exec gridlabd -D SCHED_PATH=$SCHED_PATH -D USE_FNCS -D METRICS_FILE=SGIP1a_metrics.json SGIP1b.glm &> gridlabd1a.log &)
(export FNCS_CONFIG_FILE=SGIP1b_substation.yaml && export FNCS_FATAL=YES && exec python3 -c "import tesp_support.original.substation_f as tesp;tesp.substation_loop_f('SGIP1b_agent_dict.json','SGIP1a',flag='NoMarket')" &> substation1a.log &)
(export FNCS_CONFIG_FILE=pypower.yaml && export FNCS_FATAL=YES && export FNCS_LOG_STDOUT=yes && exec python3 -c "import tesp_support.original.tso_PYPOWER_f as tesp;tesp.tso_pypower_loop_f('sgip1_pp.json','SGIP1a')" &> pypower1a.log &)
(export FNCS_FATAL=YES && export FNCS_LOG_STDOUT=yes && export WEATHER_CONFIG=SGIP1b_weather.json && exec python3 -c "import tesp_support.weather.weather_agent_f as tesp;tesp.startWeatherAgent('weather.dat')" &> weather1a.log &)
