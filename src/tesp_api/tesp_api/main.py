# Copyright (C) 2019-2022 Battelle Memorial Institute
# file: main.py

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import sqlite3

from entity import assign_defaults
from entity import assign_item_defaults
from entity import Entity
from model import GLModel
from modifier import GLMModifier

from store import entities_path
from store import feeders_path


class mytest:
    def test(self):
        return

def test_resample():
    np.random.seed(0)
    # create DataFrame with hourly index
    df = pd.DataFrame(index=pd.date_range('2020-01-06', '2020-01-07', freq='h'))
    # add column to show sales by hour
    df['sales'] = np.random.randint(low=0, high=20, size=len(df.index))
    df.to_csv("time_series_test1.csv")
    minute_df = df['sales'].resample('10T').interpolate()
    minute_df.to_csv("time_series_test1_1Minute.csv")


def fredtest():
    testMod = GLMModifier()
    testMod.model.read(feeders_path + "R1-12.47-1.glm")
    loads = testMod.get_objects('load')
    meter_counter = 0
    house_counter = 0
    house_meter_counter = 0
    for obj_id in loads.instance:
        #add meter for this load
        meter_counter = meter_counter + 1
        meter_name = 'meter_' +str(meter_counter)
        meter = testMod.add_object('meter', meter_name, [])
        meter['parent'] = obj_id
        #how much power is going to be needed
        #while kva < total_kva:
        house_meter_counter = house_meter_counter + 1
        #add parent meter for houses to follow
        house_meter_name = 'house_meter_' + str(house_meter_counter)
        meter = testMod.add_object('meter', house_meter_name, [])
        meter['parent'] = meter_name
        #add house
        house_counter = house_counter + 1
        house_name = 'house_' + str(house_counter)
        house = testMod.add_object('house', house_name, [])
        house['parent'] = house_meter_name
    testMod.write_model("test.glm")


def _test1():
    mylist = {}
    # entity_names = ["SimulationConfig", "BackboneFiles",  "WeatherPrep", "FeederGenerator",
    #             "EplusConfiguration", "PYPOWERConfiguration", "AgentPrep", "ThermostatSchedule"]
    # entity_names = ['house', 'inverter', 'battery', 'object solar', 'waterheater']

    try:
        conn = sqlite3.connect(entities_path + 'test.db')
        print("Opened database successfully")
    except:
        print("Database Sqlite3.db not formed")

    # this a config  -- file probably going to be static json
    file_name = entities_path + 'feeder_defaults.json'
    myEntity = mytest()
    assign_defaults(myEntity, file_name)
    name = 'rgnPenResHeat'
    print(name)
    print(type(myEntity.__getattribute__(name)))
    print(myEntity.__getattribute__(name))

    # this a config  -- file probably going to be static
    myEntity = mytest()
    assign_item_defaults(myEntity, file_name)
    print(myEntity.rgnPenResHeat.datatype)
    print(myEntity.rgnPenResHeat.item)
    print(myEntity.rgnPenResHeat)
    print(myEntity.rgnPenResHeat.value)

    # Better to use Entity as subclass like substation for metrics
    # Better to use Entity as object models for editing and persistence like glm_model,

    # this a multiple config file a using dictionary list persistence
    file_name = 'glm_objects.json'
    with open(entities_path + file_name, 'r', encoding='utf-8') as json_file:
        entities = json.load(json_file)
        mylist = {}
        for name in entities:
            mylist[name] = Entity(name, entities[name])
            print(mylist[name].toHelp())
            mylist[name].instanceToSQLite(conn)


def _test2():
    # Test model.py
    model_file = GLModel()
    tval = model_file.read(feeders_path + "R1-12.47-1.glm")
    # Output json with new parameters
    model_file.write(entities_path + "test.glm")
    model_file.instancesToSQLite()

    print(model_file.entitiesToHelp())
    print(model_file.instancesToGLM())

    op = open(entities_path + 'glm_objects2.json', 'w', encoding='utf-8')
    json.dump(model_file.entitiesToJson(), op, ensure_ascii=False, indent=2)
    op.close()


def _test3():
    modobject = GLMModifier()

    tval = modobject.read_model(feeders_path + "R1-12.47-1.glm")

    for name in modobject.model.entities:
        print(modobject.model.entities[name].toHelp())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # _test1()
    # _test2()
    # _test3()
    fredtest()
    # test_resample()
