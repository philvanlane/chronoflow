## Setup

To be able to run these notebooks, make sure you have installed all required dependencies as outlined in the main `README.md` file.

Additionally, you need to make sure you have the `ChronoFlow` code (i.e. the `/chronoflow/ChronoFlow.py` file) stored locally! The easiest way to do this is by cloning this entire repository, and updating the second line in each tutorial notebook:

```sys.path.append('/...path_to_chronoflow_repo/chronoflow/')```

to the appropriate path in your local environment (_note that this needs to point to the **chronoflow** subfolder within the main ChronoFlow repository_). That way, your computer will know where to look when running ```from ChronoFlow import *```.
