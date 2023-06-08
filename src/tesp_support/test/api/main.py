# Copyright (C) 2019-2022 Battelle Memorial Institute
# file: main.py

# This is a sample Python script.
import tesp_support.api.parse_helpers as p
import tesp_support.api.entity as e
import tesp_support.api.model as m
import tesp_support.api.modifier as mf
import tesp_support.api.store as s
import tesp_support.api.gridpiq as q
import feeder_generator_demo as fgd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    p._test()
    e._test()
    m._test1()
    m._test2()
    mf._test1()
    mf._test2()
    s._test_debug_resample()
    s._test_csv()
    s._test_sqlite()
    s._test_read()
    s._test_dir()
    q._test()
    fgd.test()

