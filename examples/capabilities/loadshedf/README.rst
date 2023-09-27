
This directory contains Python and Java versions of a
loadshed example on the 13-bus IEEE test feeder, modeled
in GridLAB-D. It differs from the other examples, in
not using the *tesp_support* Python package. Instead, three
local source files have been provided as possible starting
points in developing your own source files in Python or Java:

- *loadshed.py* is a loadshedding agent, implemented in Python for FNCS. Test with *run.sh*. A local copy of *fncs.py* is provided, so this example runs without importing the *tesp_support* package.
- *loadshed.java* is the same loadshedding agent, implemented in Java for FNCS. Test with *runjava.sh*
- *plot_loadshed.py* is a plotting program for the simulation results
- *kill5570.sh* is a helper script that shuts down a TESP federation
- *loadshed.glm* is the IEEE 13-bus model for GridLAB-D, configured to run as a FNCS or HELICS federate with various options
- *loadshed_dict.json* supports plotting the GridLAB-D metrics
- *killtesp.sh* is an alternative method of shutting down the TESP federation by process ID (pid) rather than by the port (5570 or 23404), Currently, only *runhpy.sh* captures pids to a file for use by this script.


loadshed - verify GridLAB-D and Python over FNCS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Results should be identical to other versions of this example.
Before running, the *clean* command will remove any existing results.

::

 cd ~/tesp/examples/loadshed
 ./clean.sh
 ./run.sh
 python3 plot_loadshed.py loadshed


loadshed - verify GridLAB-D and Java over FNCS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Results should be identical to other versions of this example.
Before running, the *clean* command will remove any existing results.

::

 cd ~/tesp/examples/loadshed
 ./clean.sh
 ./runjava.sh
 python3 plot_loadshed.py loadshed


If you're interested in C++ agent development, the GitHub
source code for *eplus_json*, under *tesp/src/energyplus*,
may provide a useful starting point. *eplus_json* handles
time synchronization, FNCS publish, FNCS subscribe, and
generating JSON output files with metrics.

Copyright (c) 2017-2023 Battelle Memorial Institute

License: https://github.com/pnnl/tesp/blob/main/LICENSE


