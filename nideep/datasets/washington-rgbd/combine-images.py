import argparse
import cv2
import numpy as np
import os
import shutil
import re
import pandas as pd


class WashingtonRGB:
    ROOT_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset'
    CSV_DEFAULT = '/mnt/raid/data/ni/dnn/pduy/rgbd-dataset/rgbd-dataset.csv'
    TYPES = {'rgb': 'crop', 'depth': 'depthcrop', 'location': 'loc', 'mask': 'maskcrop'}

    def __init__(self, root_dir='', output=''):
        self.root_dir = root_dir if root_dir != '' else self.ROOT_DEFAULT
        self.output = output

    # Load the dataset metadata to a Pandas dataframe and save the result to a csv file if it did not exists
    def load_metadata(self, csv_output=''):
        if os.path.exists(self.CSV_DEFAULT):
            return pd.read_csv(self.CSV_DEFAULT)

        else:
            file_list = os.walk(self.root_dir)

            data = []

            for current_root, dirs, files in file_list:
                files = [f for f in files if 'mask' not in f and 'loc' not in f and 'pose' not in f]

                for f in files:
                    pose_value = -1

                    name_components = f.split('_')
                    # reverse the list to void the cases when the category name has an "_"
                    reversed_name_components = np.flip(name_components, axis=0)

                    if len(reversed_name_components) < 5:
                        continue

                    category = reversed_name_components[4]
                    instance_number = reversed_name_components[3]
                    video_no = reversed_name_components[2]
                    frame_no = reversed_name_components[1]
                    data_type = reversed_name_components[0].split('.')[0]

                    if int(frame_no) % 5 == 1:
                        name_components[len(name_components) - 1] = 'pose.txt'
                        pose_file_name = '_'.join(name_components)

                        try:
                            with open(current_root + '/' + pose_file_name, 'r') as pose_file:
                                pose_value = pose_file.readline()
                                print "pose value = " + str(pose_value)
                        except IOError as e:
                            print "I/O error({0}): {1}".format(e.errno, e.strerror)

                    print "processing " + f

                    data.append({'location':  current_root + '/' + f,
                                 'category': category,
                                 'instance_number': int(instance_number),
                                 'video_no': int(video_no),
                                 'frame_no': int(frame_no),
                                 'pose': float(pose_value),
                                 'data_type': data_type})

            data_df = pd.DataFrame(data)

            # generate csv file name and save file
            output_components = self.root_dir.split('/')
            filename = output_components[len(output_components) - 1] \
                if output_components[len(output_components) - 1] != '' \
                else output_components[len(output_components) - 2]

            if csv_output == '':
                file_path = self.root_dir + '/' + filename + '.csv'
            else:
                file_path = csv_output + '/' + filename + '.csv'

            print "file name: " + filename + '.csv'
            data_df.to_csv(file_path, index=False)

        return data_df

    # Joining the rgb image with the corresponding depth map in to 1 image. Left: RGB, Right: Depth map
    def combine_rgb_with_depth(self, root, files, output_path):
        if len(files) != 0:
            files = [f for f in files if 'mask' not in f and 'loc' not in f]
            files = np.sort(files)
            indices = np.linspace(0, len(files), num=len(files) / 2 + 1, endpoint=True)

            for index in indices:
                i = int(index)
                try:
                    left_img = cv2.imread(root + "/" + files[i])
                    right_img = cv2.imread(root + "/" + files[i + 1])
                    img = np.concatenate((left_img, right_img), axis=1)
                    cv2.imwrite(output_path + '/'
                                + files[i].replace('.png', '_')
                                + files[i + 1], img)
                except (IndexError, ValueError):
                    pass

    def extract_rgb_only(self, output_path):
        data_df = self.load_metadata()
        rgb_files = data_df[data_df['data_type'] == 'crop']['location']

        for f in rgb_files:
            shutil.copy(self.root_dir + "/" + f, output_path)

    def combine_rgb_with_angle(self, angle, output_path):
        data_frame = self.load_metadata()
        rotated_files = np.sort(data_frame[(data_frame['frame_no'] == int(angle)) & (data_frame['data_type'] == 'crop')]['location'])
        original_files = np.sort(data_frame[(data_frame['frame_no'] == 1) & (data_frame['data_type'] == 'crop')]['location'])

        pairs = zip(original_files, rotated_files)

        for pair in pairs:
            left_img_name = pair[0].split('/')[len(pair[0].split('/')) - 1]
            right_img_name = pair[1].split('/')[len(pair[1].split('/')) - 1]

            print "processing " + left_img_name + " and " + right_img_name

            left_img = cv2.imread(pair[0])
            right_img = cv2.imread(pair[1])

            smaller_height = min(len(left_img), len(right_img))
            smaller_width = min(len(left_img[0]), len(right_img[0]))

            left_img = left_img[0:smaller_height, 0:smaller_width, :]
            right_img = right_img[0:smaller_height, 0:smaller_width, :]

            img = np.concatenate((left_img, right_img), axis = 1)
            cv2.imwrite(output_path + '/' \
                        + left_img_name.replace('.png', '_') \
                        + right_img_name, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", default='')
    parser.add_argument("--output", default='')
    parser.add_argument("--angle", default=10)
    parser.add_argument("--csv_input", default='')
    args = parser.parse_args()

    if args.output != '' and not os.path.exists(args.output):
        os.makedirs(args.output)

    # file_list = os.walk(args.rootdir)
    washington_dataset = WashingtonRGB(args.rootdir, args.csv_input)
    data_df = washington_dataset.load_metadata(args.rootdir)
    # combine_rgb_with_angle(data_df, args.angle, args.output)

    # for root, dirs, files in file_list:
        # combine_rgb_with_depth(root, files, parser.output)
        # extract_rgb_only(root, files, args.output)
        # combine_rgb_with_angle(root, files, 10, args.output)

