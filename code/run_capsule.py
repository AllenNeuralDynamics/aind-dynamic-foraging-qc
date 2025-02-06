import logging
import csv
import json
import numpy as np
import glob
from pathlib import Path
from datetime import datetime, timezone
import pytz
import matplotlib.pyplot as plt
from aind_log_utils.log import setup_logging
from aind_data_schema.core.quality_control import (
    QCEvaluation,
    QCMetric,
    QCStatus,
    Stage,
    Status,
    QualityControl,
)
from aind_data_schema_models.modalities import Modality


def Bool2Status(boolean_value, t=None):
    """Convert a boolean value to a QCStatus object."""
    if boolean_value:
        return QCStatus(
            evaluator="Automated", status=Status.PASS, timestamp=t.isoformat()
        )
    else:
        return QCStatus(
            evaluator="Automated", status=Status.FAIL, timestamp=t.isoformat()
        )


def load_json_file(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: {file_path} not found.")


def create_evaluation(
    name,
    description,
    metrics,
    modality=Modality.BEHAVIOR,
    stage=Stage.RAW,
    allow_failed=False,
):
    """Create a QC evaluation object."""
    return QCEvaluation(
        name=name,
        modality=modality,
        stage=stage,
        metrics=metrics,
        allow_failed_metrics=allow_failed,
        description=description,
    )


def calculate_lick_intervals(behavior_json):
    right = behavior_json['B_RightLickTime']
    left = behavior_json['B_LeftLickTime']
    threshold = .1 # time in ms to consider as a fast interval
    same_side_l = np.diff(left)
    same_side_r = np.diff(right)
    if len(right) > 0:
        # calculate left interval and fraction
        same_side_l_frac = round(np.mean(same_side_l <= threshold), 4)
        LeftLickIntervalPercent = same_side_l_frac * 100
    if len(left) > 0:
        # calculate right interval and fraction
        same_side_r_frac = round(np.mean(same_side_r <= threshold), 4)
        RightLickIntervalPercent = same_side_r_frac * 100
    if len(right) > 0 and len(left) > 0:
        # calculate same side lick interval and fraction for both right and left
        same_side_combined = np.concatenate([same_side_l, same_side_r])
        same_side_frac = round(np.mean(same_side_combined <= threshold), 4)
        # calculate cross side interval and frac
        right_dummy = np.ones(np.shape(right))  # array used to assign lick direction
        left_dummy = np.negative(np.ones(np.shape(left)))
        # 2d arrays pairing each time with a 1 (right) or -1 (left)
        stacked_right = np.column_stack((right_dummy, right))
        stacked_left = np.column_stack((left_dummy, left))
        # concatenate stacked_right and stacked_left then sort based on time element
        # e.g. [[-1, 10], [1, 15], [-1, 20], [1, 25]...]. Ones added to assign lick side to times
        merged_sorted = np.array(sorted(np.concatenate((stacked_right, stacked_left)),
                            key=lambda x: x[1]))
        diffs = np.diff(merged_sorted[:, 0])    # take difference of 1 (right) or -1 (left)
        # take difference of next index with previous at indices where directions are opposite
        cross_sides = np.array([merged_sorted[i + 1, 1] - merged_sorted[i, 1] for i in np.where(diffs != 0)])[0]
        cross_side_frac = round(np.mean(cross_sides <= threshold), 4)
        CrossSideIntervalPercent = cross_side_frac * 100
        results = {
            'LeftLickIntervalPercent':LeftLickIntervalPercent,
            'RightLickIntervalPercent':RightLickIntervalPercent,
            'SameSideIntervalPercent':same_side_frac *100, 
            'CrossSideIntervalPercent':CrossSideIntervalPercent
        }
        return results


def main():
    # Paths and setup
    base_path = Path("/data/test_raw_data_2")
    results_folder = Path("../results")
    results_folder.mkdir(parents=True, exist_ok=True)
    qc_folder = Path("../results/aind-dynamic-foraging-qc")
    qc_folder.mkdir(parents=True, exist_ok=True)

    # Load JSON files
    subject_data = load_json_file(base_path / "subject.json")
    subject_id = subject_data.get("subject_id")
    if not subject_id:
        logging.error("Error: Subject ID is missing from subject.json.")

    data_disc_json = load_json_file(base_path / "data_description.json")
    asset_name = data_disc_json.get("name")
    setup_logging("aind-dynamic-foraging-qc", mouse_id=subject_id, session_name=asset_name)

    # Load behavior JSON
    # Regex pattern is <subject_id>_YYYY-MM-DD_HH-MM-SS.json
    pattern = "/data/fiber_raw_data/behavior/[0-9]*_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"
    matching_behavior_files = glob.glob(pattern)
    if matching_behavior_files:
        behavior_json = load_json_file(matching_behavior_files[0])
    else:
        logging.info("NO BEHAVIOR JSON")

    # Create evaluations with our timezone
    seattle_tz = pytz.timezone("America/Los_Angeles")
    evaluations = []

    if 'drop_frames_tag' in behavior_json:
        logging.info('Running dropped frames check')
        evaluations.append(
            create_evaluation(
                "dropped frames check", 
                "pass when there are no dropped frames", 
                [QCMetric(
                    name="dropped frames",
                    value = behavior_json['drop_frames_tag'],
                    status_history=[
                        Bool2Status(
                            behavior_json['drop_frames_tag']==0, t=datetime.now(seattle_tz)
                        )
                    ]
                )],
                modality=Modality.BEHAVIOR_VIDEOS
            )
        )
        # If we have dropped frames, then cameras will be listed here with their recorded frames
        # iterate through each camera and report the number of dropped frames
        for camera in behavior_json['frame_num']:
            diff = behavior_json['trigger_length'] - behavior_json['frame_num'][camera]
            logging.info('Running dropped frames check for camera {}'.format(camera))
            evaluations.append(
                create_evaluation(
                    "dropped frames for each camera", 
                    "pass when there are no dropped frames", 
                    [QCMetric(
                        name="dropped frames for camera {}".format(camera),
                        value = diff,
                        status_history=[
                            Bool2Status(
                                diff==0, t=datetime.now(seattle_tz)
                            )
                        ]
                    )],
                    modality=Modality.BEHAVIOR_VIDEOS
                )
            )
    else:
       logging.info('SKIPPING dropped frames check, no drop_frames_tag')
            
    if 'Experimenter' in behavior_json:
        logging.info('Running check for researcher name')
        evaluations.append(
            create_evaluation(
                "Check researcher name", 
                "pass when researcher name is not default name", 
                [QCMetric(
                    name="researcher name",
                    value = behavior_json['Experimenter'],
                    status_history=[
                        Bool2Status(
                            behavior_json['Experimenter']!='the ghost in the shell', t=datetime.now(seattle_tz)
                        )
                    ]
                )]
            )
        )
    else:
        logging.info('SKIPPING check for researcher name, no Experimenter')

    if 'dirty_files' in behavior_json:
        logging.info('Running check for untracked file changes')
        evaluations.append(
            create_evaluation(
                "Check for untracked file changes", 
                "pass when no dirty files where in the code repository", 
                [QCMetric(
                    name="untracked local changes",
                    value = behavior_json['dirty_files'],
                    status_history=[
                        Bool2Status(
                            not behavior_json['repo_dirty_flag'], t=datetime.now(seattle_tz)
                        )
                    ]
                )]
            )
        )
    else:
        logging.info('SKIPPING check for untracked file changes, no dirty_files')

    # Check side bias
    if 'B_Bias' in behavior_json:
        logging.info('Running bias check')
        mean_bias = np.mean(behavior_json['B_Bias'])
        max_bias = behavior_json['B_Bias'][np.argmax(np.abs(behavior_json['B_Bias']))]
        evaluations.append(
            create_evaluation(
                "Side bias", 
                "pass when average bias is less than 0.75", 
                [QCMetric(
                    name="average side bias",
                    value = mean_bias,
                    status_history=[
                        Bool2Status(
                            np.abs(mean_bias) < 0.75, t=datetime.now(seattle_tz)
                        )
                    ]
                ),
                QCMetric(
                    name="Max side bias",
                    value = max_bias,
                    status_history=[
                        Bool2Status(
                            np.abs(max_bias) < 1, t=datetime.now(seattle_tz)
                        )
                    ]
                )
                ]
            )
        )
    else:
        logging.info('SKIPPING bias check, no B_Bias')

    # Check side bias
    if ('B_LeftLickTime' in behavior_json) and ('B_RightLickTime' in behavior_json):
        logging.info('Running lick interval check')
        intervals = calculate_lick_intervals(behavior_json)
        evaluations.append(
            create_evaluation(
                "Lick Intervals", 
                "pass when lick intervals <100ms are less than 10 percent of licks", 
                [QCMetric(
                    name="Left Lick Interval (%)",
                    value = intervals['LeftLickIntervalPercent'],
                    status_history=[
                        Bool2Status(
                            intervals['LeftLickIntervalPercent'] < 10, t=datetime.now(seattle_tz)
                        )
                    ]
                ),
                QCMetric(
                    name="Right Lick Interval (%)",
                    value = intervals['RightLickIntervalPercent'],
                    status_history=[
                        Bool2Status(
                            intervals['RightLickIntervalPercent'] < 10, t=datetime.now(seattle_tz)
                        )
                    ]
                ),
                QCMetric(
                    name="Cross Side Lick Interval (%)",
                    value = intervals['CrossSideIntervalPercent'],
                    status_history=[
                        Bool2Status(
                            intervals['CrossSideIntervalPercent'] < 10, t=datetime.now(seattle_tz)
                        )
                    ]
                )]
            )
        )
    else:
        logging.info('SKIPPING lick interval check')

    logging.info('Running minimum trial check')
    evaluations.append(
        create_evaluation(
            "Check for minimum trials", 
            "pass when at least 50 trials were performed", 
            [QCMetric(
                name="Number of completed trials",
                value = behavior_json.get('BS_FinisheTrialN',0),
                status_history=[
                    Bool2Status(
                        behavior_json.get('BS_FinisheTrialN',0) > 50, t=datetime.now(seattle_tz)
                    )
                ]
            )]
        )
    )

    # Create QC object and save
    qc = QualityControl(evaluations=evaluations)
    qc.write_standard_file(output_directory=str(results_folder))
    # We'd like to have our files organized such that QC is in the 
    # results directory while plots are in a named folder.
    # This allows the final results asset to have the same structure
    # We need to generate QC in the parent to ensure it works with the 
    # Web portal 
    excluded_file = "quality_control.json"
    # Iterate over files in the results directory
    for filename in os.listdir(results_folder):
        source_path = os.path.join(results_folder, filename)
        destination_path = os.path.join(qc_folder, filename)

        # Move everything except the excluded file
        if os.path.isfile(source_path) and filename != excluded_file:
            shutil.move(source_path, destination_path)

if __name__ == "__main__":
    main()
