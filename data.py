import xml.etree.ElementTree as ET
from datetime import datetime

## Loading in the data from data/experiments/experiment_1, which contains button_presses.csv for the labels, and other csv files for the features. In experiments_1/meta we have the system time and the device info
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os
import math
from tqdm import trange


def load_data(path, within_range=True, temp_features=True):
    # Load the starting time
    time_df = pd.read_csv(path + "meta/time.csv")
    start_time = time_df.loc[time_df["event"] == "START", "system time"].iloc[
        0
    ]

    data_frames = []
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            df = pd.read_csv(path + filename)
            # Check if 'Time (s)' column exists
            if "Time (s)" in df.columns:
                # Convert 'Time (s)' column to datetime index for each dataframe
                df.index = pd.to_datetime(
                    df["Time (s)"],
                    unit="s",
                    origin=pd.Timestamp(start_time, unit="s"),
                )
                data_frames.append(df)
            else:
                print(f"'Time (s)' column not found in file: {filename}")
                print(f"Columns found: {df.columns}")

    # Concatenate dataframes
    data = pd.concat(data_frames)

    # resample to 10 Hz
    data_resampled = data.resample("100ms").mean()

    # Load label dataset
    labels = pd.read_csv(
        path + "button_presses.csv", names=["Timestamp", "Label"]
    )
    labels["Timestamp"] = pd.to_datetime(labels["Timestamp"], unit="s")

    if filter:
        # Filter timestamps within label range
        first_label_timestamp = labels["Timestamp"].iloc[0]
        last_label_timestamp = labels["Timestamp"].iloc[-1]
        data_resampled = data_resampled[
            (data_resampled.index >= first_label_timestamp)
            & (data_resampled.index <= last_label_timestamp)
        ]

    if len(data_resampled):
        # Add labels
        def get_recent_label(row):
            return labels[labels["Timestamp"] <= row.name]["Label"].iloc[-1]

        data_resampled["Label"] = data_resampled.apply(
            get_recent_label, axis=1
        )
        
        if temp_features:
            # Add temporal label features
            def get_time_until_next(row):
                next_label = labels[labels["Timestamp"] > row.name][
                    "Timestamp"
                ].min()
                if pd.isnull(next_label):
                    return pd.NaT
                else:
                    return (next_label - row.name).total_seconds()

            def get_time_since_previous(row):
                previous_label = labels[labels["Timestamp"] < row.name][
                    "Timestamp"
                ].max()
                if pd.isnull(previous_label):
                    return pd.NaT
                else:
                    return (row.name - previous_label).total_seconds()

            data_resampled["Time_Until_Next_Label"] = data_resampled.apply(
                get_time_until_next, axis=1
            )
            data_resampled["Time_Since_Previous_Label"] = data_resampled.apply(
                get_time_since_previous, axis=1
            )

    return data_resampled


# Define a conversion function
def convert_timestamp(timestamp):
    datetime_obj = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    return datetime_obj


# Load the xml file into a dataframe
def load_xml(path, convert_time=False):
    # Parse the XML file
    tree = ET.parse(path + "activity_11340269258.tcx")
    root = tree.getroot()

    # Define the namespaces
    namespaces = {
        "tc": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2",
        "activity": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2",
        "ns3": "http://www.garmin.com/xmlschemas/ActivityExtension/v2",
        "ns5": "http://www.garmin.com/xmlschemas/ActivityGoals/v1",
        "ns2": "http://www.garmin.com/xmlschemas/UserProfile/v2",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "ns4": "http://www.garmin.com/xmlschemas/ProfileExtension/v1",
    }

    # Extract data from XML and create a dictionary
    xml_data = {"Time": [], "AltitudeMeters": [], "HeartRate": []}

    for trackpoint in root.findall(".//tc:Trackpoint", namespaces):
        time = trackpoint.find("tc:Time", namespaces).text
        altitude = trackpoint.find("tc:AltitudeMeters", namespaces).text
        heart_rate = trackpoint.find(
            "tc:HeartRateBpm/tc:Value", namespaces
        ).text

        xml_data["Time"].append(time)
        xml_data["AltitudeMeters"].append(altitude)
        xml_data["HeartRate"].append(heart_rate)

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(xml_data)

    df['AltitudeMeters'] = df['AltitudeMeters'].astype(float)
    df['HeartRate'] = df['HeartRate'].astype(float)

    # Apply the conversion function to the 'Time' column
    if convert_time:
        df["Time"] = df["Time"].apply(convert_timestamp)

    df = df.set_index("Time")

    return df


def merge(data, xml_data):
    first_timestamp = data.index[0]
    last_timestamp = data.index[-1]
    df_filtered = xml_data[
        (xml_data.index >= first_timestamp)
        & (xml_data.index <= last_timestamp)
    ]
    merged_df = pd.merge(
        data, df_filtered, left_index=True, right_index=True, how="left"
    )

    return merged_df


def state_transition(x, w, dt):
    return x + dt * w


def observation_model(x):
    return [
        math.sin(x[0]),
        -math.cos(x[0]) * math.sin(x[1]),
        -math.cos(x[0]) * math.cos(x[1]),
    ]


def jacobian_state_transition():
    return np.eye(3)


def jacobian_observation(x):
    return np.array(
        [
            [math.cos(x[0]), 0],
            [
                math.sin(x[0]) * math.sin(x[1]),
                -math.cos(x[0]) * math.cos(x[1]),
            ],
            [math.sin(x[0]) * math.cos(x[1]), math.cos(x[0]) * math.sin(x[1])],
        ]
    )


if __name__ == "__main__":
    experiment = "data/experiments/experiment_2/"
    # data = load_data(experiment)
    # xml_data = load_xml(experiment)
    # df = merge(data, xml_data)

    df = pd.read_csv(experiment + "merged/merged.csv", index_col=0)

    # Create placeholders for roll, pitch and yaw
    roll, pitch, yaw = (
        np.zeros(df.shape[0]),
        np.zeros(df.shape[0]),
        np.zeros(df.shape[0]),
    )

    for i, (_, row) in enumerate(df.iterrows()):
        if (
            row[
                [
                    "X (rad/s)",
                    "Y (rad/s)",
                    "Z (rad/s)",
                    "X (m/s^2)",
                    "Y (m/s^2)",
                    "Z (m/s^2)",
                ]
            ]
            .isnull()
            .any()
        ):
            continue
        ax, ay, az = row["X (m/s^2)"], row["Y (m/s^2)"], row["Z (m/s^2)"]
        gx, gy, gz = row["X (rad/s)"], row["Y (rad/s)"], row["Z (rad/s)"]

        # Calculate pitch and roll using formulas provided in previous answers
        pitch[i] = math.atan2(ay, az)
        roll[i] = math.atan2(-ax, math.sqrt(ay**2 + az**2))

        # Compute the yaw by integrating gyroscope data
        if (
            i == 0
        ):  # if it's the first observation, there's no previous yaw and time
            yaw[i] = 0
        else:
            yaw[i] = yaw[i - 1] + gz * (
                row["Time (s).1"] - df.iloc[i - 1]["Time (s).1"]
            )  # integrate using the trapezoidal rule

    # Convert radians to degrees
    roll, pitch, yaw = np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    # Add roll, pitch, and yaw to the DataFrame
    df["Roll (°)"] = roll
    df["Pitch (°)"] = pitch
    df["Yaw (°)"] = yaw
    df.to_csv(experiment + "merged/added_features.csv")
