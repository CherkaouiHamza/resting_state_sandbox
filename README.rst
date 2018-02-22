Resting state sandbox
######################


Authors
-------
Hamza Cherkaoui


Synopsis
--------

Simple package to offer **ACI**, **seed based correlation** and
**seed based GLM** analysis of the resting state acquisition done with the PET
MR scanner in Service Hospitalier Frederic Joliot.


Dependencies
------------

* nilearn  
* pypreprocess  


Configuration
-------------

Please edit the *config.ini* file to specify the desired subject to analyze and
the data directory.

WARNING: If you do not properly set the data directory you will not be able to
run the analysis.

Please update your .bashrc by adding

.. code-block:: bash

    export PYTHONPATH="/path/to/resting_state_sandbox/:$PYTHONPATH"

And then source your .bashrc

.. code-block:: bash

    source ~/.bashrc


Instructions
------------

Launch the example:

.. code-block:: bash

    python seed_glm.py; python seed_corr.py; python ica.py

You can then inspect the preprocessing report:

.. code-block:: bash

    firefox pypreprocess_output/report_preproc.html

To watch the produce results, please check the `analysis_output` directory:
