# aind-dynamic-foraging-qc (ver 0.1)

QC capsule for dynamic foraging behavior (modality:behavior) raw data acquired together with HARP/Bonsai-based behavior 

run_capsule.py : main script

Following the "alternate-workflow" with which you don't need to make a new asset, instead directly pushing QC.json to DocDB.

https://github.com/AllenNeuralDynamics/aind-qc-portal?tab=readme-ov-file#alternate-workflow

___
**QC tests implemented**

- Check for dropped frames in cameras
    - If dropped frames exist, report number dropped for each camera
- Check that experimenter name is not the default name
- Add QC alert for dirty files
- check for side bias (extreme bias < 1, and average bias < .75)
- [ ] should add check for lick interval less than 50ms

    
___
**Steps:**

1.reading rawdata

2.generating figures and metrics

3.submitting figures to kachery to obtain unique url

4.Composing QC/QCevals/QCmetrics

5.Pushing QC,josn to DocDB

6.visualizing, manual QCing under AIND-QCportal-app

https://qc.allenneuraldynamics.org/qc_portal_app

