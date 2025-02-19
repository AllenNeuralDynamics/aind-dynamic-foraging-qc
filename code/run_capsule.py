import logging
import csv
import json
import os
import glob
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
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
    right = behavior_json["B_RightLickTime"]
    left = behavior_json["B_LeftLickTime"]
    threshold = 0.1  # time in ms to consider as a fast interval
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
        merged_sorted = np.array(
            sorted(np.concatenate((stacked_right, stacked_left)), key=lambda x: x[1])
        )
        diffs = np.diff(
            merged_sorted[:, 0]
        )  # take difference of 1 (right) or -1 (left)
        # take difference of next index with previous at indices where directions are opposite
        cross_sides = np.array(
            [
                merged_sorted[i + 1, 1] - merged_sorted[i, 1]
                for i in np.where(diffs != 0)
            ]
        )[0]
        cross_side_frac = round(np.mean(cross_sides <= threshold), 4)
        CrossSideIntervalPercent = cross_side_frac * 100
        results = {
            "LeftLickIntervalPercent": LeftLickIntervalPercent,
            "RightLickIntervalPercent": RightLickIntervalPercent,
            "SameSideIntervalPercent": same_side_frac * 100,
            "CrossSideIntervalPercent": CrossSideIntervalPercent,
        }
        return results

def plot_bias(behavior_json,results_folder):
    plt.figure()
    plt.xlabel('Time from first go cue(s)')
    plt.ylabel('Side Bias')
    plt.axhline(+0.7,color='r', linestyle='--')
    plt.axhline(-0.7,color='r', linestyle='--')
    plt.axhline(0,color='k', linestyle='--')
    plt.ylim([-1,+1]) 

    if len(behavior_json['B_Bias']) == len(behavior_json['B_GoCueTime']):
        plt.plot(np.array(behavior_json['B_GoCueTime'])-behavior_json['B_GoCueTime'][0],behavior_json['B_Bias'],'k',linewidth=2)
        start = 0
        stop = behavior_json['B_GoCueTime'][-1] - behavior_json['B_GoCueTime'][0]
        plt.xlim([start,stop])
    else:
        plt.plot(behavior_json['B_Bias'],'k',linewidth=2)

    plt.savefig(f"{results_folder}/side_bias.png", dpi=300, bbox_inches="tight")

def main():
    # Paths and setup
    base_path = Path("/data/fiber_raw_data")
    results_folder = Path("../results")
    results_folder.mkdir(parents=True, exist_ok=True)
    qc_folder = Path("../results/qc-raw")
    qc_folder.mkdir(parents=True, exist_ok=True)

    ref_folder = Path("qc-raw")

    # Load JSON files
    subject_data = load_json_file(base_path / "subject.json")
    subject_id = subject_data.get("subject_id")
    if not subject_id:
        logging.error("Error: Subject ID is missing from subject.json.")

    data_disc_json = load_json_file(base_path / "data_description.json")
    asset_name = data_disc_json.get("name")
    setup_logging(
        "aind-dynamic-foraging-qc", mouse_id=subject_id, session_name=asset_name
    )

    session_json = load_json_file(base_path / "session.json")

 
    # Load behavior JSON
    # Regex pattern is <subject_id>_YYYY-MM-DD_HH-MM-SS.json
    pattern = "/data/fiber_raw_data/behavior/[0-9]*_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9].json"
    matching_behavior_files = glob.glob(pattern)
    if matching_behavior_files:
        behavior_json = load_json_file(matching_behavior_files[0])
    else:
        logging.info("NO BEHAVIOR JSON")

    # Create bias plot
    plot_bias(behavior_json,results_folder)

    # Create evaluations with our timezone
    seattle_tz = pytz.timezone("America/Los_Angeles")
    evaluations = []

    if "drop_frames_tag" in behavior_json:
        logging.info("Running dropped frames check")
        camera_metrics = []
        # If we have dropped frames, then cameras will be listed here with their recorded frames
        # iterate through each camera and report the number of dropped frames
        frame_metric = QCMetric(
            name="dropped frames",
            value=behavior_json["drop_frames_tag"],
            status_history=[
                Bool2Status(
                    behavior_json["drop_frames_tag"] == 0, t=datetime.now(seattle_tz),
                )
            ],
        )
        for camera in behavior_json["frame_num"]:
            diff = behavior_json["trigger_length"] - behavior_json["frame_num"][camera]
            logging.info("Running dropped frames check for camera {}".format(camera))
            camera_metrics.append(
                QCMetric(
                    name="dropped frames for camera {}".format(camera),
                    value=diff,
                    status_history=[Bool2Status(diff == 0, t=datetime.now(seattle_tz))],
                )
            )
        evaluations.append(
            create_evaluation(
                "dropped frames check",
                "pass when there are no dropped frames",
                [frame_metric, *camera_metrics],
                modality=Modality.BEHAVIOR_VIDEOS,
            )
        )
    else:
        logging.info("SKIPPING dropped frames check, no drop_frames_tag")

    if ("Experimenter" in behavior_json) and ("dirty_files" in behavior_json):
        logging.info("Running check for basic configuration")
        evaluations.append(
            create_evaluation(
                "Basic Configuration",
                "pass when researcher name is not default name, and code repo is clean",
                [
                    QCMetric(
                        name="researcher name",
                        value=behavior_json["Experimenter"],
                        description='Experimenter name should not be set to the default value of "the ghost in the shell"',
                        status_history=[
                            Bool2Status(
                                behavior_json["Experimenter"]
                                != "the ghost in the shell",
                                t=datetime.now(seattle_tz),
                            )
                        ],
                    ),
                    QCMetric(
                        name="untracked local changes",
                        description='Whether the code base had untracked changes, and if so, which files',
                        value=behavior_json["dirty_files"],
                        status_history=[
                            Bool2Status(
                                not behavior_json["repo_dirty_flag"],
                                t=datetime.now(seattle_tz),
                            )
                        ],
                    )
                ],
            )
        )
    else:
        logging.info("SKIPPING check for basic configuration")


    # Check side bias
    if "B_Bias" in behavior_json:
        logging.info("Running bias check")
        mean_bias = np.mean(behavior_json["B_Bias"])
        max_bias = behavior_json["B_Bias"][np.argmax(np.abs(behavior_json["B_Bias"]))]
        evaluations.append(
            create_evaluation(
                "Side bias",
                "pass when average bias is less than 0.75",
                [
                    QCMetric(
                        name="average side bias",
                        value=mean_bias,
                        status_history=[
                            Bool2Status(
                                np.abs(mean_bias) < 0.75, t=datetime.now(seattle_tz)
                            )
                        ],
                        reference=str(ref_folder / "side_bias.png")
                    ),
                    QCMetric(
                        name="Max side bias",
                        value=max_bias,
                        status_history=[
                            Bool2Status(
                                np.abs(max_bias) < 1, t=datetime.now(seattle_tz)
                            )
                        ],
                        reference=str(ref_folder / "side_bias.png")
                    ),
                ],
            )
        )
    else:
        logging.info("SKIPPING bias check, no B_Bias")

    # Check side bias
    if ("B_LeftLickTime" in behavior_json) and ("B_RightLickTime" in behavior_json):
        logging.info("Running lick interval check")
        intervals = calculate_lick_intervals(behavior_json)
        evaluations.append(
            create_evaluation(
                "Lick Intervals",
                "pass when lick intervals <100ms are less than 10 percent of licks",
                [
                    QCMetric(
                        name="Left Lick Interval (%)",
                        value=intervals["LeftLickIntervalPercent"],
                        status_history=[
                            Bool2Status(
                                intervals["LeftLickIntervalPercent"] < 10,
                                t=datetime.now(seattle_tz),
                            )
                        ],
                    ),
                    QCMetric(
                        name="Right Lick Interval (%)",
                        value=intervals["RightLickIntervalPercent"],
                        status_history=[
                            Bool2Status(
                                intervals["RightLickIntervalPercent"] < 10,
                                t=datetime.now(seattle_tz),
                            )
                        ],
                    ),
                    QCMetric(
                        name="Cross Side Lick Interval (%)",
                        value=intervals["CrossSideIntervalPercent"],
                        status_history=[
                            Bool2Status(
                                intervals["CrossSideIntervalPercent"] < 10,
                                t=datetime.now(seattle_tz),
                            )
                        ],
                    ),
                ],
            )
        )
    else:
        logging.info("SKIPPING lick interval check")

    logging.info("Running session length check")
    if ('stimulus_epochs' in session_json) and ('stimulus_start_time' in session_json['stimulus_epochs']):
        stimulus_start = session_json['stimulus_epochs'][0]['stimulus_start_time']
        stimulus_end = session_json['stimulus_epochs'][0]['stimulus_end_time']
        session_length = datetime.fromisoformat(stimulus_end) - datetime.fromisoformat(stimulus_start) 
    else:
        session_length = timedelta(minutes=0)

    evaluations.append(
        create_evaluation(
            "Session Length Check",
            "pass when at least 50 trials were performed, and stimulus epoch was at least 10 minutes",
            [
                QCMetric(
                    name="Number of completed trials",
                    description='Must complete at least 50 trials to pass',
                    value=behavior_json.get("BS_FinisheTrialN", 0),
                    status_history=[
                        Bool2Status(
                            behavior_json.get("BS_FinisheTrialN", 0) > 50,
                            t=datetime.now(seattle_tz),
                        )
                    ],
                ),
                QCMetric(
                    name="Length of stimulus epoch",
                    description="Must be at least 10 minutes",
                    value=session_length,
                    status_history=[
                        Bool2Status(
                            session_length > timedelta(minutes=10),
                            t=datetime.now(seattle_tz),
                        )
                    ],
                )
            ],
        )
    )

    # TODO - move files?

    # Create QC object and save
    qc = QualityControl(evaluations=evaluations)
    qc.write_standard_file(output_directory=str(results_folder))


if __name__ == "__main__":
    main()
