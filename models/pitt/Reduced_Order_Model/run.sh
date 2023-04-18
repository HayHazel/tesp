(export FNCS_BROKER="tcp://*:5570" && export FNCS_FATAL=YES && exec fncs_broker 5 &> broker.log &)
(export FNCS_CONFIG_FILE=eplus.yaml && export FNCS_FATAL=YES && exec energyplus -w FL-Miami_Intl_Ap.epw -d output -r SchoolDualController.idf &> eplus.log &)
(export FNCS_CONFIG_FILE=eplus_json.yaml && export FNCS_FATAL=YES && exec eplus_agent 432000s 300s SchoolDualController eplus_TE_metrics.json 0.02 25 4 4 &> eplus_json.log &)
(export FNCS_FATAL=YES && exec gridlabd -D USE_FNCS -D METRICS_FILE=TE_metrics.json TE.glm &> gridlabd.log &)
(export FNCS_CONFIG_FILE=TE_substation.yaml && export FNCS_FATAL=YES && exec python -c "import tesp_support.substation as tesp;tesp.substation_loop('TE_agent_dict.json','TE')" &> substation.log &)
(export FNCS_CONFIG_FILE=pypower.yaml && export FNCS_FATAL=YES && export FNCS_LOG_STDOUT=yes && exec python -c "import tesp_support.tso_PYPOWER_f as tesp;tesp.pypower_loop_f('TE_pp.json','TE')" &> pypower.log &)
