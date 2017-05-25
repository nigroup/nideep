import os
import numpy as np
import cv2
import argparse
import shutil
import pandas as pd
import logging
from sklearn.model_selection import train_test_split


class WashingtonRGBD(object):
    """
    
    Data Wrapper class for WashingtonRGBD dataset
    Attributes
    -----------
    root_dir: root directory until the rgbd-dataset folder. For example: /mnt/raid/data/ni/dnn/pduy/rgbd-dataset
    csv_default: the default directory for loading/saving the csv description of the dataset
    csv_interpolated_default: the default directory for loading/saving the pose-interpolated csv description of the 
    dataset.
    
    """

    def __init__(self, root_dir='', csv_default='', csv_perframe_default='', csv_interpolated_default=''):
        self.logger = logging.getLogger(__name__)
        self.root_dir = root_dir
        self.csv_default = csv_default
        self.csv_perframe_default = csv_perframe_default
        self.csv_interpolated_default = csv_interpolated_default

    # Load the dataset metadata to a Pandas dataframe and save the result to a csv file
    # if it does not exists
    # otherwise read the csv
    # The missing pose values will be saved as -1
    def load_metadata(self):
        if os.path.isfile(self.csv_default):
            self.logger.info('reading from ' + self.csv_default)
            return pd.read_csv(self.csv_default)

        file_list = os.walk(self.root_dir)

        data = []

        for current_root, _, files in file_list:
            # For the time being, we do not work on the mask, location
            # For the pose, it should be attached to the corresponding data entry, not as a separate entry
            # files = [f for f in files if 'mask' not in f and 'loc' not in f and 'pose' not in f]
            files = [f for f in files if 'pose' not in f]

            for f in files:
                self.logger.info("processing " + f)

                pose_value = -1

                name_components = f.split('_')

                # The category name can be 1 word or 2 words, such as 'apple' or 'cell_phone'
                # So, when splitting the file name by '_', there can be 5 or 6 components
                # That's why I read the name backward to make sure I get the proper data pieces
                # reversed_name_components = np.flip(name_components, axis=0)

                if len(name_components) < 5:
                    continue

                n_components = len(name_components)
                if n_components > 5:    # if n_components > 5, it means the category name has more than 1 word
                    category = '_'.join(name_components[0: n_components - 4])
                else:
                    category = name_components[0]

                instance_number = name_components[-4]
                video_no = name_components[-3]
                frame_no = name_components[-2]
                data_type = name_components[-1].split('.')[0]

                name_components[n_components - 1] = 'pose.txt'
                pose_file_name = '_'.join(name_components)

                try:
                    with open(os.path.join(current_root, pose_file_name), 'r') as pose_file:
                        pose_value = pose_file.readline()
                        self.logger.info("pose value = " + str(pose_value))
                except IOError:
                    self.logger.info("No pose value for this instance!")

                data.append({'location': os.path.join(current_root, f),
                             'category': category,
                             'instance_number': int(instance_number),
                             'video_no': int(video_no),
                             'frame_no': int(frame_no),
                             'pose': float(pose_value),
                             'data_type': data_type})

        data_frame = pd.DataFrame(data) \
            .sort_values(['data_type', 'category', 'instance_number', 'video_no', 'frame_no'])

        self.logger.info("csv saved to file: " + self.csv_default + '.csv')
        data_frame.to_csv(self.csv_default, index=False)

        return data_frame

    # Interpolate the missing pose values (saved as -1 by the load_metadata() method)
    def interpolate_poses(self, data_frame):
        if os.path.isfile(self.csv_interpolated_default):
            self.logger.info('reading from ' + self.csv_interpolated_default)
            return pd.read_csv(self.csv_interpolated_default)

        self.logger.info('Interpolating ...')

        sorted_df = data_frame.sort_values(['data_type', 'category', 'instance_number', 'video_no', 'frame_no'])
        poses = np.array(sorted_df['pose'])

        current_video = -1

        for i in range(0, len(poses)):
            if (sorted_df['video_no'][i] != current_video) and (poses[i] == 0):
                unit_diff_angle = poses[i + 5] / 5

            if poses[i] == -1:
                poses[i] = poses[i - 1] + unit_diff_angle
                if poses[i] > 360:
                    poses[i] = poses[i] - 360

        sorted_df['pose'] = poses
        sorted_df.to_csv(self.csv_interpolated_default, index=False)

        self.logger.info('Interpolation finished!')

        return sorted_df

    # Get a new dataframe where each row represent all information about 1 frame including the rgb and depth locations
    # structure: ['category', 'instance_number', 'video_no', 'frame_no', 'crop_location', 'depthcrop_location']
    def get_df_per_frame(self):
        if os.path.isfile(self.csv_perframe_default):
            return pd.read_csv(self.csv_perframe_default)

        raw_df = self.interpolate_poses(self.load_metadata())
        raw_rgb_df = raw_df[raw_df.data_type == 'crop']
        data = []

        for i in range(len(raw_rgb_df.index)):
            current_row = raw_df.iloc[[i]]
            current_category = current_row['category'].values[0]
            current_instance_number = current_row['instance_number'].values[0]
            current_video_no = current_row['video_no'].values[0]
            current_frame_no = current_row['frame_no'].values[0]
            current_pose = current_row['pose'].values[0]
            current_crop_location = current_row['location'].values[0]
            current_depthcrop_location = raw_df[(raw_df.category == current_category)
                                                & (raw_df.instance_number == current_instance_number)
                                                & (raw_df.video_no == current_video_no)
                                                & (raw_df.frame_no == current_frame_no)
                                                & (raw_df.data_type == 'depthcrop')]['location'].values[0]

            self.logger.info("processing " + os.path.split(current_crop_location)[1]
                             + " and " + os.path.split(current_depthcrop_location)[1])

            data.append({
                'category': current_category,
                'instance_number': current_instance_number,
                'video_no': current_video_no,
                'frame_no': current_frame_no,
                'pose': current_pose,
                'crop_location': current_crop_location,
                'depthcrop_location': current_depthcrop_location
            })

        new_df = pd.DataFrame(data)
        new_df.to_csv(self.csv_perframe_default, index=False)
        return new_df

    def extract_rgb_only(self, output_path):
        data_frame = self.load_metadata()
        rgb_files = data_frame[data_frame['data_type'] == 'crop']['location']

        for f in rgb_files:
            shutil.copy(os.path.join(self.root_dir, f), output_path)

    # Combine an rgb image with a rotated image of the same object horizontally into 1 image,
    # together with a train-test-split for doing a hold-out validation.
    # Only one elevation video_no is taken, the other elevations are ignored
    # Left:RGB, (Middle: Depth Map),  Right: Rotation
    def combine_viewpoints(self, angle, video_no, should_include_depth, output_path):

        def join(df, output_path):
            for i in range(len(df.index)):
                current_original_file_df = df.iloc[[i]]

                # Filtering out the rotation candidates,
                # most of the things should be the same, except for frame_no,
                # and the 2 poses should differentiate by the provided angle with an error bound of +-1
                rotation_candidates = df[(df.category == current_original_file_df.category.values[0])
                                         & (df.instance_number == current_original_file_df.instance_number.values[0])
                                         & (df.video_no == current_original_file_df.video_no.values[0])
                                         & (df.pose <= current_original_file_df.pose.values[0] + angle + 1)
                                         & (df.pose >= current_original_file_df.pose.values[0] + angle - 1)]

                for j in range(len(rotation_candidates.index)):
                    current_rotated_file_df = rotation_candidates.iloc[[j]]

                    locations = []
                    names = []

                    locations.append(current_original_file_df.crop_location.values[0])
                    names.append(os.path.split(locations[0])[1])

                    if should_include_depth:
                        locations.append(current_original_file_df.depthcrop_location.values[0])
                        names.append(os.path.split(locations[1])[1])

                    locations.append(current_rotated_file_df.crop_location.values[0])
                    names.append(os.path.split(locations[2])[1])

                    self.logger.info("merging " + " and ".join(names))
                    self.perform_cv_combination(locations, names, output_path)

        data_frame = self.get_df_per_frame()

        # Filter out one elevation only
        data_frame = data_frame[data_frame['video_no'] == video_no]

        # train test split
        train, test = train_test_split(data_frame, test_size=0.2)

        # construct training and test sets, saving to disk
        join(train, os.path.join(output_path, 'train'))
        join(test, os.path.join(output_path, 'test'))

    # this method combines every rgb frame with its depthmap on the right
    def combine_rgb_depth(self, output_path):
        def join(df, output_path):
            for i in range(len(df.index)):
                current_row = df.iloc[[i]]
                locations = [current_row.crop_location.values[0], current_row.depthcrop_location.values[0]]
                names = [os.path.split(location)[1] for location in locations]

                self.perform_cv_combination(locations, names, output_path)

        df = self.get_df_per_frame()

        train, test = train_test_split(df, test_size=0.2)

        join(train, os.path.join(output_path, 'train'))
        join(test, os.path.join(output_path, 'test'))

    # combine all of the image in an array to only 1 image and save to file
    @staticmethod
    def perform_cv_combination(locations, names, output_path):
        if output_path != '' and not os.path.isdir(output_path):
            os.makedirs(output_path)

        imgs = [cv2.imread(location) for location in locations]

        min_height = min([len(img) for img in imgs])
        min_width = min([len(img[0]) for img in imgs])

        imgs = map(lambda x: cv2.resize(x, (min_width, min_height)), imgs)

        img = np.concatenate(imgs, axis=1)
        cv2.imwrite(os.path.join(output_path,
                                 '_'.join([os.path.splitext(name)[0] for name in names])
                                 + os.path.splitext(names[0])[1]),
                    img)


if __name__ == '__main__':
    ROOT_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset'
    CSV_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/rgbd-dataset.csv'
    CSV_PERFRAME_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/rgbd-dataset-perframe.csv'
    CSV_INTERPOLATED_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/rgbd-dataset-interpolated.csv'

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", default=ROOT_DEFAULT)
    parser.add_argument("--csv_dir", default=CSV_DEFAULT)
    parser.add_argument("--csv_perframe_dir", default=CSV_PERFRAME_DEFAULT)
    parser.add_argument("--csv_interpolated_dir", default=CSV_INTERPOLATED_DEFAULT)
    parser.add_argument("--processed_data_output", default='')
    parser.add_argument("--angle", default=10, type=int)
    parser.add_argument("--depth_included", default=False, type=bool)

    args = parser.parse_args()

    if args.processed_data_output != '' and not os.path.isdir(args.processed_data_output):
        os.makedirs(args.processed_data_output)

    washington_dataset = WashingtonRGBD(root_dir=args.rootdir,
                                        csv_default=args.csv_dir,
                                        csv_perframe_default=args.csv_perframe_dir,
                                        csv_interpolated_default=args.csv_interpolated_dir)

    washington_dataset.load_metadata()

    # washington_dataset.combine_viewpoints(angle=args.angle,
    #                                       video_no=1,
    #                                       should_include_depth=args.depth_included,
    #                                       output_path=args.processed_data_output)

    # washington_dataset.combine_rgb_depth(args.processed_data_output)

