from torchvision import transforms
from transforms import CropBottom


LAYOUT = {
    "Training Plots": {
        "MSE": ["Multiline", ["MSE/train", "MSE/test"]],
        "RMSE": ["Multiline", ["RMSE/train", "RMSE/test"]],
    },
}

FEATURE_COLUMS = ['PSM1_joint_1', 'PSM1_joint_2', 'PSM1_joint_3', 'PSM1_joint_4',
                  'PSM1_joint_5', 'PSM1_joint_6', 'PSM1_jaw_angle', 'PSM1_ee_x',
                  'PSM1_ee_y', 'PSM1_ee_z', 'PSM1_Orientation_Matrix_[1,1]',
                  'PSM1_Orientation_Matrix_[1,2]', 'PSM1_Orientation_Matrix_[1,3]',
                  'PSM1_Orientation_Matrix_[2,1]', 'PSM1_Orientation_Matrix_[2,2]',
                  'PSM1_Orientation_Matrix_[2,3]', 'PSM1_Orientation_Matrix_[3,1]',
                  'PSM1_Orientation_Matrix_[3,2]', 'PSM1_Orientation_Matrix_[3,3]',
                  'PSM2_joint_1', 'PSM2_joint_2', 'PSM2_joint_3', 'PSM2_joint_4',
                  'PSM2_joint_5', 'PSM2_joint_6', 'PSM2_jaw_angle', 'PSM2_ee_x',
                  'PSM2_ee_y', 'PSM2_ee_z', 'PSM2_Orientation_Matrix_[1,1]',
                  'PSM2_Orientation_Matrix_[1,2]', 'PSM2_Orientation_Matrix_[1,3]',
                  'PSM2_Orientation_Matrix_[2,1]', 'PSM2_Orientation_Matrix_[2,2]',
                  'PSM2_Orientation_Matrix_[2,3]', 'PSM2_Orientation_Matrix_[3,1]',
                  'PSM2_Orientation_Matrix_[3,2]', 'PSM2_Orientation_Matrix_[3,3]']

IMAGE_COLUMS = ['ZED Camera Left', 'ZED Camera Right']

TIME_COLUMN = ["Time (Seconds)"]

VELOCITY_COLUMNS = \
    [f'PSM{nr}_ee_v_{axis}' for axis in ['x', 'y', 'z'] for nr in [1, 2]] \
    + [f'PSM{nr}_joint_{joint}_v' for joint in range(1, 7) for nr in [1, 2]] \
    + [f'PSM{nr}_jaw_angle_v' for nr in [1, 2]]

ACCELERATION_COLUMNS = \
    [f'PSM{nr}_ee_a_{axis}' for axis in ['x', 'y', 'z'] for nr in [1, 2]] \
    + [f'PSM{nr}_joint_{joint}_a' for joint in range(1, 7) for nr in [1, 2]] \
    + [f'PSM{nr}_jaw_angle_a' for nr in [1, 2]]

TARGET_COLUMNS = ['Force_x_smooth', 'Force_y_smooth', 'Force_z_smooth']

START_END_TIMES = {
    "force_policy": {
        1: [(0, -1)],
        2: [(0, -1)],
        # 3: [(800, -1)],
        # 4: [(400, 1200), (2500, -1)],
        3: [(0, -1)],
        4: [(0, -1)],
        6: [(2000, -1)],
        8: [(500, -1)],
        9: [(700, 1700), (2200, -1)],
        10: [(1700, -1)],
        11: [(500, -1)]
    },
    "no_force_policy": {
        1: [(500, -1)],
        3: [(700, -1)],
        4: [(500, -1)]
    }
}

EXCEL_FILE_NAMES = {
    "force_policy": {
        key: (f"dec6_force_no_TA_lastP_randomPosHeight_cs100_run{key}.xlsx" if 1 <= key <= 15)
            #   f"dec19_force_no_TA_lastP_randomPosHeight_cs100_run{key}.xlsx" if 16 <= key <= 30 else
            #   f"dec20_force_no_TA_lastP_randomPosHeight_cs100_run{key}.xlsx")
        for key in range(1, 15)
    },
    # "no_force_policy": {
    #     key: (f"dec6_no_force_no_TA_lastP_randomPosHeight_cs100_run{key}.xlsx" if 1 <= key <= 15 else
    #           f"dec19_no_force_no_TA_lastP_randomPosHeight_cs100_run{key}.xlsx" if 16 <= key <= 30 else
    #           f"dec20_no_force_no_TA_lastP_randomPosHeight_cs100_run{key}.xlsx")
    #     for key in range(1, 51)
    # },
}


NUM_IMAGE_FEATURES = 30
NUM_ROBOT_FEATURES = 58
NUM_ROBOT_FEATURES_INCL_ACCEL = 78
CNN_MODEL_VERSION = "efficientnet_v2_m"


# Transformer Config
SEQ_LENGTH = 10
HIDDEN_LAYERS = [128, 256]
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT_RATE = 0.3

DEFAULT_TEST_RUNS = [[13, 29, 33, 36, 39, 45], []]

RES_NET_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

RES_NET_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    CropBottom((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

ENCODER_WEIGHTS_FN = "encoder_weights.pth"
TARGET_SCALER_FN = "transformations/target_scaler.joblib"
FEATURE_SCALER_FN = "transformations/feature_scaler.joblib"

MOVING_AVG_WINDOW_SIZE = 5