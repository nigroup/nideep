from washington_rgbd import *
import numpy as np
from matplotlib import cm
from scipy.misc import *

logger = logging.getLogger(__name__)


def tile_border(rgb_image, dst_size):
    old_height = rgb_image.shape[0]
    old_width = rgb_image.shape[1]

    if old_height > old_width:
        rgb_image = rgb_image.transpose(1, 0, 2)

    height = rgb_image.shape[0]
    width = rgb_image.shape[1]

    new_height = int(height * dst_size * 1.0 / width)
    rgb_image = cv2.resize(rgb_image, (dst_size, new_height))
    tiling_size = int((dst_size - new_height) * 1.0 / 2)

    first_row_matrix = np.tile(rgb_image[0, :, :], (tiling_size, 1, 1)) if len(rgb_image.shape) > 2 \
        else np.tile(rgb_image[0, :], (tiling_size, 1))

    last_row_matrix = np.tile(rgb_image[new_height - 1, :, :], (dst_size - new_height - tiling_size, 1, 1)) \
        if len(rgb_image.shape) > 2 \
        else np.tile(rgb_image[new_height - 1, :], (dst_size - new_height - tiling_size, 1))

    rgb_image = np.concatenate([first_row_matrix,
                                rgb_image,
                                last_row_matrix],
                               axis=0)

    if old_height > old_width:
        rgb_image = rgb_image.transpose(1, 0, 2)

    return rgb_image


def colorize_depth(depth_map):
    # scale everything to [0, 255]
    sorted_depth = np.unique(np.sort(depth_map.flatten()))
    sorted_depth = np.delete(sorted_depth, 0)
    min_depth = sorted_depth[0]
    max_depth = sorted_depth[len(sorted_depth) - 1]

    # for i in range(depth_map.shape[0]):
    #     for j in range(depth_map.shape[1]):
    #         pixel = depth_map[i][j]
    #         depth_map[i][j] = (pixel - min_depth) * 1.0 / (max_depth - min_depth) if pixel >= min_depth else 0

    depth_map = np.asarray(map(lambda pixel:
                               (pixel - min_depth) * 1.0 / (max_depth - min_depth),
                               depth_map))

    # Apply jet colormap to it
    depth_map = np.uint8(cm.jet_r(depth_map) * 255)
    return depth_map[:, :, 0:3]


def preprocess_frame(row, file_dir):
    image = imread(row.crop_location)
    depth = imread(row.depthcrop_location, flatten=True)

    image = tile_border(image, 256)
    depth = tile_border(colorize_depth(depth), 256)

    combined_image = np.concatenate([image, depth], axis=1)
    cv2.imwrite(file_dir, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))


def build_training_data(washington_df, output_base_path):
    if not os.path.isdir(output_base_path):
        os.mkdir(output_base_path)

    categories = np.sort(np.unique(washington_df.category))

    train_df = []
    for i in range(washington_df.shape[0]):
        row = washington_df.iloc[i]
        label_vector = np.array([int(c == row.category) for c in categories])

        file_path = os.path.join(output_base_path,
                                 '_'.join([str(row.category),
                                           str(row.instance_number),
                                           str(row.video_no),
                                           str(row.frame_no)])
                                 + ".png")

        logger.info('processing ' + file_path)

        train_df.append({'location': file_path,
                         'label': label_vector,
                         'category': row.category,
                         'instance_number': row.instance_number,
                         'video_no': row.video_no,
                         'frame_no': row.frame_no})

        preprocess_frame(row, file_path)

    train_df = pd.DataFrame(train_df)
    train_df.to_csv(os.path.join(output_base_path, 'train_info.csv'), index=False)


def train_test_split_5_frames(train_info_df, output_base_path):
    categories = np.unique(train_info_df.category)

    test_df = pd.DataFrame(columns=list(train_info_df))
    train_df = pd.DataFrame(columns=list(train_info_df))
    for category in categories:
        # Select 1 category for test and insert the whole category to test set
        category_df = train_info_df[train_info_df.category == category]
        max_instance = np.max(category_df.instance_number)
        test_instance = np.random.randint(max_instance) + 1
        test_df = test_df.append(category_df[category_df.instance_number == test_instance])

        # For the rest, select every 5th frame for test, rest for training
        training_instances_df = category_df[category_df.instance_number != test_instance]
        train_df = train_df.append(training_instances_df[training_instances_df.frame_no % 5 != 1])
        test_df = test_df.append(training_instances_df[training_instances_df.frame_no % 5 == 1])

    if train_df.shape[0] + test_df.shape[0] == train_info_df.shape[0]:
        logger.info('saving test and train to csv')
        train_df.to_csv(os.path.join(output_base_path, 'training_set.csv'), index=False)
        test_df.to_csv(os.path.join(output_base_path, 'test_set.csv'), index=False)
    else:
        logger.warning('Splitting error')
        logger.warning('training size = %d' % train_df.shape[0])
        logger.warning('testing size = %d' % test_df.shape[0])


if __name__ == '__main__':
    ROOT_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset'
    CSV_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/rgbd-dataset.csv'
    CSV_AGGREGATED_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/rgbd-dataset-interpolated-aggregated.csv'
    CSV_INTERPOLATED_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/rgbd-dataset-interpolated.csv'
    PROCESSED_PAIR_PATH = '/mnt/raid/data/ni/dnn/pduy/eitel-et-al-data/'

    logging.basicConfig(level=logging.INFO)

    washington_dataset = WashingtonRGBD(root_dir=ROOT_DEFAULT,
                                        csv_default=CSV_DEFAULT,
                                        csv_perframe_default=CSV_AGGREGATED_DEFAULT,
                                        csv_interpolated_default=CSV_INTERPOLATED_DEFAULT)

    aggregate_washington_df = washington_dataset.aggregate_frame_data()
    small_df = aggregate_washington_df[(aggregate_washington_df.category == 'apple')
                                       | (aggregate_washington_df.category == 'keyboard')
                                       | (aggregate_washington_df.category == 'banana')]

    # build_training_data(aggregate_washington_df, PROCESSED_PAIR_PATH)
    train_test_split_5_frames(pd.read_csv(os.path.join(PROCESSED_PAIR_PATH, 'train_info.csv'), index_col=False),
                              PROCESSED_PAIR_PATH)
