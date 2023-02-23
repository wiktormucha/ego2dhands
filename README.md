# ego2dhands

List of files:
-datasets/augumentations.py -> Own functions for image augumentations, in the end not emplyed due to usage of albumentations
-datasets/FreiHAND.py -> Data loader for FreiHand dataset
-datasets/h2o.py -> Data loader for H2O dataset

-models/backbones.py -> File with backbones networks
-models/heads.py -> File with prediction head
-models/models.py -> File with complete models (backbone+head)

-utils/egocentric.py -> File with complete models (backbone+head)
-utils/hand_detector.py -> Hand detector related functions
-utils/metrics.py -> Metrics for evaluation
-utils/testing.py -> Functions for testing
-utils/trainer.py -> Training class
-utils/utils.py -> Useful basic functions

-config.py -> File with configurations
-egocentric_demo.py ->  Runs prdicion on egocentric given frame
-test_all_models.py -> Tests all models on FreiHAND dataset
-test_ego.py -> Runs egocentric predictions tests and saves metrics to .csv files
-train.py -> Main file that runs training.