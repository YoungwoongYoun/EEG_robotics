"""Canonical channel definitions shared by preprocessing and experiments."""

EEG_CHANNELS_22 = (
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
)

MI9_CHANNELS = (
    "FC1",
    "FCz",
    "FC2",
    "C3",
    "Cz",
    "C4",
    "CP1",
    "CPz",
    "CP2",
)

MI9_INDICES = tuple(EEG_CHANNELS_22.index(channel) for channel in MI9_CHANNELS)

CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")

TRAIN_CUE_EVENT_IDS = {
    "769": 1,
    "770": 2,
    "771": 3,
    "772": 4,
}
EVALUATION_CUE_EVENT_ID = {"783": 1}
REJECTED_TRIAL_EVENT_ID = {"1023": 1}
