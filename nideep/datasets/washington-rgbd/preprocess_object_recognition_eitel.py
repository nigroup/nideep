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
    # sorted_depth = np.delete(sorted_depth, 0)
    min_depth = sorted_depth[0]
    max_depth = sorted_depth[len(sorted_depth) - 1]

    depth_map = np.asarray(map(lambda pixel:
                               (pixel - min_depth) * 1.0 / (max_depth - min_depth),
                               depth_map))

    # Apply jet colormap to it
    depth_map = np.uint8(cm.jet_r(depth_map) * 255)
    return depth_map[:, :, 0:3]


def preprocess_frame(row, file_dir):
    try:
        image = imread(row['crop_location'])
        # depth = imread(row['depthcrop_location'], flatten=True)   # Using the original depth
        depth = imread(row['filled_depthcrop_location'], flatten=True)     # Using the filled depth

        image = tile_border(image, 256)
        depth = tile_border(colorize_depth(depth), 256)

        combined_image = np.concatenate([image, depth], axis=1)
        cv2.imwrite(file_dir, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
    except IOError:
        print 'file ' + row.crop_location + ' does not exist. Probably in the other train split half'


def build_training_data(washington_df, save_path):
    dir_path = os.path.split(save_path)[0]
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    categories = np.sort(np.unique(washington_df.category))

    train_df = []
    for i in range(washington_df.shape[0]):
        row = washington_df.iloc[i]
        label_vector = np.array([int(c == row.category) for c in categories])

        file_path = os.path.join(dir_path,
                                 '_'.join([str(row.category),
                                           str(int(row.instance_number)),
                                           str(int(row.video_no)),
                                           str(int(row.frame_no))])
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
    train_df.to_csv(save_path, index=False)


def mapping_to_gan_data(data_frame, gan_image_dir, saving_dir, need_preprocess=False):
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)

    new_df = []
    for i in range(data_frame.shape[0]):
        current_row = data_frame.iloc[i]

        location = current_row.location
        label = current_row.label
        category = current_row.category
        instance_number = current_row.instance_number
        video_no = current_row.video_no
        frame_no = current_row.frame_no

        basic_name_element = '_'.join([category,
                                       str(int(instance_number)),
                                       str(int(video_no)),
                                       str(int(frame_no))])

        if need_preprocess:
            rgb_file_name_crop = '_'.join([basic_name_element,
                                           'crop'])
            rgb_file_name_depthcrop = '_'.join([basic_name_element,
                                                'depthcrop'])
            # rgb_file_name = '_'.join([rgb_file_name_crop, rgb_file_name_depthcrop]) + '-inputs.png'    #OLD NAMING
            # depth_file_name = '_'.join([rgb_file_name_crop, rgb_file_name_depthcrop]) + '-outputs.png' #OLD NAMING
            rgb_file_name = rgb_file_name_crop + '-inputs.png'
            depth_file_name = rgb_file_name_crop + '-outputs.png'
            row = {'crop_location': os.path.join(gan_image_dir, rgb_file_name),
                   'filled_depthcrop_location': os.path.join(gan_image_dir, depth_file_name)}

            preprocess_frame(pd.Series(row), os.path.join(saving_dir, basic_name_element + '.png'))

        new_df.append({
            'location': location,
            'label': label,
            'category': category,
            'instance_number': instance_number,
            'video_no': video_no,
            'frame_no': frame_no,
            'location_generated': os.path.join(saving_dir, basic_name_element + '.png')
        })

    new_df = pd.DataFrame(new_df)
    new_df.to_csv(os.path.join(saving_dir, 'gan-test-data.csv'), index=False)


def preprocess_a_folder(folder_dir, output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    file_list = os.walk(folder_dir)

    for current_dir, _, files in file_list:
        files = np.sort(files)
        for i in range(len(files)):
            f = files[i]
            if 'inputs' in f:
                crop_location = os.path.join(current_dir, f)
                depthcrop_location = os.path.join(current_dir, files[i + 1])
                row = {'crop_location': crop_location,
                       'depthcrop_location': depthcrop_location}

                preprocess_frame(row, os.path.join(output_path, f))


if __name__ == '__main__':
    ROOT_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset'
    CSV_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/rgbd-dataset.csv'
    CSV_AGGREGATED_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/rgbd-dataset-interpolated-aggregated.csv'
    CSV_INTERPOLATED_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/rgbd-dataset-interpolated.csv'
    PROCESSED_PAIR_PATH = '/mnt/raid/data/ni/dnn/pduy/eitel-et-al-data/'
    GAN_TEST_FOLDER_50 = '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-50-test/images'
    GAN_PROCESSED_FOLDER_75 = '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-50-test/processed-images'
    GAN_TEST_FOLDER_25 = '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-25-test/images'
    GAN_PROCESSED_FOLDER_25 = '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-25-test/processed-images'
    GAN_TEST_FOLDER_10 = '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-10-test/images'
    GAN_PROCESSED_FOLDER_10 = '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-10-test/processed-images'
    GAN_TEST_FOLDER_10_40EP = '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-10-40ep-test/images'
    GAN_PROCESSED_FOLDER_10_40EP = '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-10-40ep-test/' \
                                   'processed-images'

    GAN_TEST_FOLDER_30_EPOCS = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset-rgb-depth-train-split-30-epochs/images/'
    GAN_PROCESSED_FOLDER_30_EPOCS = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset-rgb-depth-train-split-30-epochs' \
                                    '/processed_images/'
    GAN_TEST_FOLDER_35_EPOCS = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset-rgb-depth-train-split-35-epochs/images/'
    GAN_PROCESSED_FOLDER_35_EPOCS = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset-rgb-depth-train-split-35-epochs' \
                                    '/processed_images/'

    logging.basicConfig(level=logging.INFO)

    # washington_dataset = WashingtonRGBD(root_dir=ROOT_DEFAULT,
    #                                     csv_default=CSV_DEFAULT,
    #                                     csv_perframe_default=CSV_AGGREGATED_DEFAULT,
    #                                     csv_interpolated_default=CSV_INTERPOLATED_DEFAULT)

    # aggregate_washington_df = washington_dataset.aggregate_frame_data()
    # washington_train_df, washington_test_df = washington_dataset.train_test_split_eitel(aggregate_washington_df)

    # small_df = aggregate_washington_df[(aggregate_washington_df.category == 'apple')
    #                                    | (aggregate_washington_df.category == 'keyboard')
    #                                    | (aggregate_washington_df.category == 'banana')]

    # build_training_data(washington_train_df, os.path.join(PROCESSED_PAIR_PATH, 'training_set.csv'))
    # build_training_data(washington_test_df, os.path.join(PROCESSED_PAIR_PATH, 'test_set.csv'))
    # train_test_split_5_frames(pd.read_csv(os.path.join(PROCESSED_PAIR_PATH, 'train_info.csv'), index_col=False),
    #                           PROCESSED_PAIR_PATH)

    train_df = pd.read_csv(os.path.join(PROCESSED_PAIR_PATH, 'training_set.csv'))
    test_df = pd.read_csv(os.path.join(PROCESSED_PAIR_PATH, 'test_set.csv'))

    mapping_to_gan_data(data_frame=train_df, gan_image_dir=GAN_TEST_FOLDER_10_40EP, saving_dir=GAN_PROCESSED_FOLDER_10_40EP,
                        need_preprocess=True)
    # preprocess_a_folder('/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-50-test/images/',
    #                     '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-50-test/processed-images/')
