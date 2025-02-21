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
    threshold = 0.05  # time in ms to consider as a fast interval
    all_licks = np.sort(left +right)
    all_diffs = np.sort(np.diff(all_licks))
    ArtifactPercent = np.mean(all_diffs < 0.0005)*100 
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
        same_side_frac = round(np.sum(same_side_combined <= threshold)/(len(right)+len(left)), 4)
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
        cross_side_frac = round(np.sum(cross_sides <= threshold)/(len(left)+len(right)), 4)
        CrossSideIntervalPercent = cross_side_frac * 100
        SameSideIntervalPercent = same_side_frac * 100
        results = {
            "LeftLickIntervalPercent": LeftLickIntervalPercent,
            "RightLickIntervalPercent": RightLickIntervalPercent,
            "SameSideIntervalPercent": SameSideIntervalPercent,
            "CrossSideIntervalPercent": CrossSideIntervalPercent,
            "ArtifactPercent": ArtifactPercent,
        }
        return results

def plot_lick_intervals(behavior_json, results_folder):
    fig, ax = plt.subplots(1,5,figsize=(8,3),sharex=True,sharey=True)

    ax[0].set_xlim(-0.01, 0.3)
    ax[0].set_title('left licks')
    ax[1].set_title('right licks')
    ax[2].set_title('left to right licks')
    ax[3].set_title('right to left licks')
    ax[4].set_title('all licks')
    ax[0].set_ylabel('counts')
    for a in ax:
        a.set_xlabel('time (s)')
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    x_values = np.linspace(-0.3, 0.3, 100)
    LeftLicksIndex=np.zeros_like(behavior_json['B_LeftLickTime'])
    RightLicksIndex=np.ones_like(behavior_json['B_RightLickTime'])
    AllLicks=np.concatenate((behavior_json['B_LeftLickTime'],behavior_json['B_RightLickTime']))
    AllLicksIndex=np.concatenate((LeftLicksIndex,RightLicksIndex))
    AllLicksSorted=np.sort(AllLicks)
    AllLicksSortedDiff=np.diff(AllLicksSorted)
    SortedIndex=np.argsort(AllLicks)
    AllLicksIndexSorted=AllLicksIndex[SortedIndex]
    AllLicksIndexSortedDiff=np.diff(AllLicksIndexSorted)
    LeftToRightLicks=AllLicksSortedDiff[AllLicksIndexSortedDiff==1]
    RightToLeftLicks=AllLicksSortedDiff[AllLicksIndexSortedDiff==-1]

    ax[0].hist(
        np.diff(behavior_json['B_LeftLickTime']), 
        bins=x_values, 
        color='red', 
        alpha=0.7,
        label='left licks'
        )
    ax[1].hist(
        np.diff(behavior_json['B_RightLickTime']), 
        bins=x_values, 
        color='blue', 
        alpha=0.7,
        label='right licks'
        )
    ax[2].hist(
        LeftToRightLicks, 
        bins=x_values, 
        color='black', 
        alpha=0.7,
        label='left to right licks'
        )
    ax[3].hist(
        RightToLeftLicks, 
        bins=x_values, 
        color='black', 
        alpha=0.7,
        label='right to left licks'
        )
    ax[4].hist(
        AllLicksSortedDiff, 
        bins=x_values, 
        color='black', 
        alpha=0.7,
        label='all licks'
        )

    plt.tight_layout()
    plt.savefig(f"{results_folder}/lick_intervals.png", dpi=300, bbox_inches="tight")

def plot_bias(behavior_json,results_folder):
    '''
        Plot a figure of the side bias, and lick spout position
        behavior_json, the data saved from the GUI
        results_folder, the place to save the figure
    '''
    fig,ax = plt.subplots(nrows=2,figsize=(6,6))
    
    # Set up side bias plot
    ax[0].set_xlabel('Trial #')
    ax[0].set_ylabel('Side Bias')
    ax[0].axhline(+0.7,color='r', linestyle='--')
    ax[0].axhline(-0.7,color='r', linestyle='--')
    ax[0].axhline(0,color='k', linestyle='--')
    ax[0].set_ylim([-1,+1]) 
    for side in ['top', 'right']:
        ax[0].spines[side].set_visible(False)
        ax[1].spines[side].set_visible(False)

    # If we have the confidence intervals, plot them
    if ('B_Bias_CI' in behavior_json):
        lower = [x[0] for x in behavior_json['B_Bias_CI']]
        upper = [x[1] for x in behavior_json['B_Bias_CI']]
        ax[0].fill_between(
            np.arange(0,len(behavior_json['B_Bias'])),
            behavior_json['B_Bias'], 
            lower, 
            upper, 
            color='gray', 
            alpha=.5
            )

    # Plot the bias trace
    if 'B_Bias' in behavior_json:
        ax[0].plot(behavior_json['B_Bias'],'k',linewidth=2)
        ax[0].set_xlim([0, len(behavior_json['B_Bias'])])

    if 'B_StagePositions' in behavior_json:
        # Extract stage positions
        if 'y1' in behavior_json['B_StagePositions'][0]:
            x = [x['x'] for x in behavior_json['B_StagePositions']]
            z = [x['z'] for x in behavior_json['B_StagePositions']]
            y1 = [x['y1'] for x in behavior_json['B_StagePositions']]
            y2 = [x['y2'] for x in behavior_json['B_StagePositions']]
        else:
            # Convert Newscale from um to mm
            x = [x['x']/1000 for x in behavior_json['B_StagePositions']]
            z = [x['z']/1000 for x in behavior_json['B_StagePositions']]
            y1 = [x['y']/1000 for x in behavior_json['B_StagePositions']]
            y2 = [x['y']/1000 for x in behavior_json['B_StagePositions']] 

        # Plot stage positions
        ax[1].plot(np.array(x)[:-1]-x[0],'r',label='X')
        ax[1].plot(np.array(y1)[:-1]-y1[0],'b',label='Y1')
        ax[1].plot(np.array(y2)[:-1]-y2[0],'lightblue',label='Y2')
        ax[1].plot(np.array(z)[:-1]-z[0],'m',label='Z')
    
        # Clean up plot
        ax[1].set_xlim([0, len(behavior_json['B_Bias'])])
        ax[1].set_xlabel('Trial #')
        ylims = ax[1].get_ylim()
        ax[1].set_ylim([np.min([-1,ylims[0]]), np.max([1,ylims[1]])])
        ax[1].set_ylabel('Lickspout Position \n relative to session start (mm)')
        ax[1].legend()          
    
    # Save figure 
    plt.savefig(f"{results_folder}/side_bias.png", dpi=300, bbox_inches="tight")

def main():
    # Paths and setup
    base_path = Path("/data/fiber_raw_data")
    results_folder = Path("../results/aind-dynamic-foraging-qc")
    results_folder.mkdir(parents=True, exist_ok=True)

    # Load JSON files
    subject_data = load_json_file(base_path / "subject.json")
    subject_id = subject_data.get("subject_id")
    if not subject_id:
        logging.error("Error: Subject ID is missing from subject.json.")

    data_disc_json = load_json_file(base_path / "data_description.json")
    asset_name = data_disc_json.get("name")
    setup_logging(
        "aind-dynamic-foraging-qc", subject_id=subject_id, asset_name=asset_name
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

    # Create lick interval plot
    plot_lick_intervals(behavior_json, results_folder)

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
                "pass when max bias is less than 1, and average side bias is less than 0.5",
                [
                    QCMetric(
                        name="average side bias",
                        description="average side bias should be less than 0.5",
                        value=mean_bias,
                        status_history=[
                            Bool2Status(
                                np.abs(mean_bias) < 0.5, t=datetime.now(seattle_tz)
                            )
                        ],
                        reference=str(results_folder / "side_bias.png")
                    ),
                    QCMetric(
                        name="Max side bias",
                        description="max side bias should be less than 1",
                        value=max_bias,
                        status_history=[
                            Bool2Status(
                                np.abs(max_bias) < 1, t=datetime.now(seattle_tz)
                            )
                        ],
                        reference=str(results_folder / "side_bias.png")
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
                        description = "% of lick intervals < 50ms. These indicate grooming bouts",
                        status_history=[
                            Bool2Status(
                                intervals["LeftLickIntervalPercent"] < 10,
                                t=datetime.now(seattle_tz),
                            )
                        ],
                        reference=str(results_folder / "lick_intervals.png")
                    ),
                    QCMetric(
                        name="Right Lick Interval (%)",
                        value=intervals["RightLickIntervalPercent"],
                        description = "% of lick intervals < 50ms. These indicate grooming bouts",
                        status_history=[
                            Bool2Status(
                                intervals["RightLickIntervalPercent"] < 10,
                                t=datetime.now(seattle_tz),
                            )
                        ],
                        reference=str(results_folder / "lick_intervals.png")
                    ),
                    QCMetric(
                        name="Cross Side Lick Interval (%)",
                        value=intervals["CrossSideIntervalPercent"],
                        description = "% of lick intervals < 50ms. These indicate grooming bouts",
                        status_history=[
                            Bool2Status(
                                intervals["CrossSideIntervalPercent"] < 10,
                                t=datetime.now(seattle_tz),
                            )
                        ],
                        reference=str(results_folder / "lick_intervals.png")
                    ),
                    QCMetric(
                        name="Artifact Percent (%)",
                        value=intervals["ArtifactPercent"],
                        description="% of lick intervals less than 0.5ms. These indicate electical artifacts",
                        status_history=[
                            Bool2Status(
                                intervals["ArtifactPercent"] < 1,
                                t=datetime.now(seattle_tz),
                            )
                        ],
                        reference=str(results_folder / "lick_intervals.png")
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
