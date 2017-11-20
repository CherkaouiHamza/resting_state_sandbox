Resting state sandbox
######################


Authors
-------
Hamza Cherkaoui


Synopsis
--------

Simple package to offer **ACI** and **seed based** analysis of the resting state
acquisition done with the PET MR scanner in Service Hospitalier Frederic Joliot.


Dependencies
------------

* nilearn  
* pypreprocess  


Configuration
-------------

Please edit the *config.py* file to specify the path to the data directory of the
acquisition: *nicolas2*, *claire5* and *claire6*.


Instructions
------------

Go to the example directory of your choice:

.. code-block:: bash

    cd resting_state_sandbox/example/nicolas2/

Launch the example:

.. code-block:: bash

    python nicolas2_resting_state.py

You can then inspect the preprocessing report:

.. code-block:: bash

    firefox pypreprocess_output/report_preproc.html

And watch the produce Defautl Mode Network:

.. code-block:: bash

    eog analysis_output/

