from washington_rgbd import *
import numpy as np
from matplotlib import cm
from scipy.misc import *

logger = logging.getLogger(__name__)


# Tile the image to a destination square image, used in Eitel et al.
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


# colorizing the depth map using jet color map
def colorize_depth(depth_map):
    # scale everything to [0, 255]
    sorted_depth = np.unique(np.sort(depth_map.flatten()))
    min_depth = sorted_depth[0]
    max_depth = sorted_depth[len(sorted_depth) - 1]

    depth_map = np.asarray(map(lambda pixel:
                               (pixel - min_depth) * 1.0 / (max_depth - min_depth),
                               depth_map))

    # Apply jet colormap to it
    depth_map = np.uint8(cm.jet_r(depth_map) * 255)
    return depth_map[:, :, 0:3]


# Given a CSV row of metadata, colorize the image and save into a destination
def preprocess_frame(row, file_dir, processing_depth=True):
    try:
        if processing_depth:
            input_img = imread(row['source_location'])
            # depth = imread(row['depthcrop_location'], flatten=True)   # Using the original depth
            target_img = imread(row['filled_depthcrop_location'], flatten=True)     # Using the filled depth

            input_img = tile_border(input_img, 256)
            target_img = tile_border(colorize_depth(target_img), 256)

        else:
            input_img = imread(row['rgb_original_path'])
            target_img = imread(row['rgb_target_path'])

            input_img = tile_border(input_img, 256)
            target_img = tile_border(target_img, 256)

        combined_image = np.concatenate([input_img, target_img], axis=1)
        cv2.imwrite(file_dir, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

    except IOError:
        logger.info('file does not exist. Probably in the other train split half')


# Build training data with labels from the washington RGBD Dataset, saving metadata to CSV
def build_training_data(washington_df, save_path):
    if os.path.exists(save_path):
        return pd.read_csv(save_path, index_col=False)

    dir_path = os.path.split(save_path)[0]
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    categories = np.sort(np.unique(washington_df.category))

    labeled_washington_df = washington_df.copy()
    labels = []
    locations = []
    for i in range(labeled_washington_df.shape[0]):
        row = labeled_washington_df.iloc[i]
        label_vector = np.array([int(c == row.category) for c in categories])

        file_path = os.path.join(dir_path,
                                 '_'.join([str(row.category),
                                           str(int(row.instance_number)),
                                           str(int(row.video_no)),
                                           str(int(row.frame_no))])
                                 + ".png")

        labels.append(label_vector)
        locations.append(file_path)

        logger.info('processing ' + file_path)

        preprocess_frame(row, file_path, processing_depth=False)

    labeled_washington_df['location'] = pd.Series(locations, index=labeled_washington_df.index)
    labeled_washington_df['label'] = pd.Series(labels, index=labeled_washington_df.index)

    labeled_washington_df.to_csv(save_path, index=False)

    return labeled_washington_df


# Create pairs of RGB - Depth image from GAN's generated data
# or RGB+Depth - RGB+Depth for the Pose-GAN (not used anymore, replaced by RGB + D)
def map_to_gan_data_with_depth(original_training_df, gan_image_dir, saving_dir, pose_preprocess=False, depth_preprocess=False,
                               need_actual_preprocess=False):
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)

    new_df = []
    for i in range(original_training_df.shape[0]):
        current_row = original_training_df.iloc[i]

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
        basic_name_element_rotated = basic_name_element + '-rotated'

        # process the data from the GAN generating depth images: RGB => Depth
        if depth_preprocess:
            logger.info('processing Depth GAN')
            rgb_file_name_crop = '_'.join([basic_name_element,
                                           'crop'])
            # rgb_file_name_depthcrop = '_'.join([basic_name_element,
            #                                     'depthcrop'])
            # rgb_file_name = '_'.join([rgb_file_name_crop, rgb_file_name_depthcrop]) + '-inputs.png'    #OLD NAMING
            # depth_file_name = '_'.join([rgb_file_name_crop, rgb_file_name_depthcrop]) + '-outputs.png' #OLD NAMING
            rgb_file_name = rgb_file_name_crop + '-inputs.png'
            depth_file_name = rgb_file_name_crop + '-outputs.png'

            if need_actual_preprocess:
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

        # process the data from the GAN generating a new pose: RGB + Depth => RGB + Depth.
        # need to process 2 frames every time
        if pose_preprocess:
            logger.info('processing Pose GAN')
            rgb_file_name_crop = '_'.join([basic_name_element,
                                           'crop'])
            # rgb_file_name = '_'.join([rgb_file_name_crop, rgb_file_name_depthcrop]) + '-inputs.png'    #OLD NAMING
            # depth_file_name = '_'.join([rgb_file_name_crop, rgb_file_name_depthcrop]) + '-outputs.png' #OLD NAMING
            input_rgb_name = rgb_file_name_crop + '-inputs.png'
            input_depth_name = rgb_file_name_crop + '-inputs_depth.png'
            output_rgb_name = rgb_file_name_crop + '-outputs.png'
            output_depth_name = rgb_file_name_crop + '-outputs_depth.png'

            if need_actual_preprocess:
                row_input = {'crop_location': os.path.join(gan_image_dir, input_rgb_name),
                             'filled_depthcrop_location': os.path.join(gan_image_dir, input_depth_name)}
                row_output = {'crop_location': os.path.join(gan_image_dir, output_rgb_name),
                              'filled_depthcrop_location': os.path.join(gan_image_dir, output_depth_name)}

                preprocess_frame(pd.Series(row_input), os.path.join(saving_dir, basic_name_element + '.png'))
                preprocess_frame(pd.Series(row_output), os.path.join(saving_dir, basic_name_element_rotated + '.png'))

            new_df.append({
                'location': location,
                'label': label,
                'category': category,
                'instance_number': instance_number,
                'video_no': video_no,
                'frame_no': frame_no,
                'location_generated': os.path.join(saving_dir, basic_name_element + '.png')
            })
            new_df.append({
                'location': location,
                'label': label,
                'category': category,
                'instance_number': instance_number,
                'video_no': video_no,
                'frame_no': frame_no,
                'location_generated': os.path.join(saving_dir, basic_name_element_rotated + '.png')
            })

    new_df = pd.DataFrame(new_df)
    new_df.to_csv(os.path.join(saving_dir, 'gan-test-data.csv'), index=False)


# Create pairs of Stereo RGBs from GAN generated data. Basically the input is concatenated with the output
def create_stereo_rgb_from_gan(original_training_df, gan_image_dir, saving_dir):
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)

    new_df = original_training_df.copy()

    locations_generated = []
    for i in range(new_df.shape[0]):
        current_row = new_df.iloc[i]

        category = current_row.category
        instance_number = current_row.instance_number
        video_no = current_row.video_no
        frame_no = current_row.frame_no

        basic_name_element = '_'.join([category,
                                       str(int(instance_number)),
                                       str(int(video_no)),
                                       str(int(frame_no))])

        rgb_file_name_crop = '_'.join([basic_name_element,
                                       'crop'])
        input_rgb_name = rgb_file_name_crop + '-inputs.png'
        output_rgb_name = rgb_file_name_crop + '-outputs.png'
        row = {'rgb_original_path': os.path.join(gan_image_dir, input_rgb_name),
               'rgb_target_path': os.path.join(gan_image_dir, output_rgb_name)}

        processed_location = os.path.join(saving_dir, basic_name_element + '.png')

        preprocess_frame(pd.Series(row), processed_location, processing_depth=False)
        locations_generated.append(processed_location)

    new_df['location_generated'] = pd.Series(locations_generated, index=new_df.index)
    new_df.to_csv(os.path.join(saving_dir, 'gan-test-data.csv'), index=False)


# Reading from a whole folder instead of from CSV, and create RGB-Depth images
def preprocess_a_folder(folder_dir, output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    file_list = os.walk(folder_dir)

    for current_dir, _, files in file_list:
        files = np.sort(files)
        for i, f in enumerate(files):
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
                                    '/pserocessed_images/'

    '''pose generation data'''
    GAN_TEST_FOLDER_POSE = '/mnt/raid/data/ni/dnn/pduy/training-pose-16bit/' \
                           'rgbd-50-reg-discrim-instance-noise-one-sided-smooth-label-filtering-categories-test/images'
    GAN_PROCESSED_FOLDER_POSE = '/mnt/raid/data/ni/dnn/pduy/training-pose-16bit/' \
                                'rgbd-50-reg-discrim-instance-noise-one-sided-smooth-label-filtering-categories-test/' \
                                'processed-images-stereo-rgb'

    PROCESSED_STEREO_RGB_PAIR_PATH = '/mnt/raid/data/ni/dnn/pduy/eitel-et-al-data-stereo-rgb/'
    GAN_TEST_FOLDER_50 = '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-50-test/images'
    CSV_EITEL_TRAIN_STEREO_RGB_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/eitel-train-stereo-rgb.csv'
    CSV_EITEL_TEST_STEREO_RGB_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/eitel-test-stereo-rgb.csv'

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

    # train_df = pd.read_csv(os.path.join(PROCESSED_PAIR_PATH, 'training_set.csv'))
    # test_df = pd.read_csv(os.path.join(PROCESSED_PAIR_PATH, 'test_set.csv'))

    # map_to_gan_data_with_depth(original_training_df=train_df, gan_image_dir=GAN_TEST_FOLDER_POSE, saving_dir=GAN_PROCESSED_FOLDER_POSE,
    #                            pose_preprocess=True)
    # preprocess_a_folder('/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-50-test/images/',
    #                     '/mnt/raid/data/ni/dnn/pduy/training-depth-16bit/rgbd-depth-50-test/processed-images/')

    train_stereo_df = pd.read_csv(CSV_EITEL_TRAIN_STEREO_RGB_DEFAULT, index_col=False)
    test_stereo_df = pd.read_csv(CSV_EITEL_TEST_STEREO_RGB_DEFAULT, index_col=False)

    processed_training_data_original_df = build_training_data(train_stereo_df,
                                                              os.path.join(PROCESSED_STEREO_RGB_PAIR_PATH,
                                                                           'training_set.csv'))
    build_training_data(test_stereo_df, os.path.join(PROCESSED_STEREO_RGB_PAIR_PATH, 'test_set.csv'))

    create_stereo_rgb_from_gan(processed_training_data_original_df, GAN_TEST_FOLDER_POSE, GAN_PROCESSED_FOLDER_POSE)

