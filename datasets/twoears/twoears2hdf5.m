function [] = twoears2hdf5(fpath, dir_dst, merge)
% TWOEARS2HDF  load twoears training data and reformat into caffe-friendly HDF5
%   TWOEARS2HDF(fpath, dir_dst) loads data from .mat file designated by fpath
%     and writes them to hdf5 files under directory dir_dst
%   Assumes:
%       rows are examples, labels are features/class columns
%       label states are -1, 0, 1 for inactive, undefined, active
%       respectively
%   The general class is only present in the scalar representation of the
%   ground truth. It is equivalent to an all-zero ground truth vector.
% 
if nargin < 3
    merge = false;
end
load(fpath, 'x', 'y', 'featureNames', 'classnames');

dir_src = fileparts(fpath);
[~, phase] = fileparts(dir_src); % test or train from directory name
assert( strcmp(phase, 'test') | strcmp(phase, 'train'), ...
    'Unable to determine phase (test vs. train).' );

% remove rows with zero entries (undefined state) in ground truth
[nozero, ~] = find( all( y~=0, 2 ) );
x2 = x( nozero, : );
y2 = y( nozero, : );
y2( y2==-1 ) = 0;
% activate general class wherever all target classes are absent
general_col = find( strcmp( classnames, 'general' ) );
y2( sum(y2, 2)==0, general_col ) = 1;

% random shuffle
o = randperm( length( y2 ) );
x2 = x2( o, : );
y2 = y2( o, : );

[x_feat, feature_type_names, y] = twoears2Blob(x2, featureNames, y2);
y_scalar = labelVec2ScalarsBlob(y);
% Afer scalarizing the ground truth, we can remove the general class column
% from the ground truths vectors.
y = removeClassFromVec(y, general_col);

if merge
    % merge all features and ground truth into same hdf5
    prefix_h5 = 'twoears_data';
    fname_h5 = sprintf('%s_%s.h5', prefix_h5, phase);
    for ii = 1 : numel(feature_type_names)
        % save formatted features to file
        if ii > 1
            write_mode = 'append';
        else
            write_mode = 'overwrite'; % first entry only
        end
        hdf5write( fullfile(dir_dst, fname_h5), ...
            strcat('/', feature_type_names{ii}), x_feat{ii}, ...
            'WriteMode', write_mode);
    end
    % append ground truth to hdf5
    hdf5write( fullfile(dir_dst, fname_h5), ...
        '/label', y, ...
        'WriteMode', 'append');
    if ~isempty(y_scalar)
        hdf5write( fullfile(dir_dst, fname_h5), ...
            '/label_scalar', y_scalar, ...
            'WriteMode', 'append');
    end
    % write hdf5 list files
    file_id = fopen( fullfile(dir_dst, sprintf('%s_%s.txt', prefix_h5, phase) ), 'w');
    fprintf(file_id, fullfile(dir_dst, fname_h5) );
    fclose(file_id);
else % everything goes into a separate hdf5
    for ii = 1 : numel(feature_type_names)
        % save formatted features to file
        fname = sprintf('feat_%s_%s.h5', feature_type_names{ii}, phase);
        hdf5write( fullfile(dir_dst, fname), strcat('/', feature_type_names{ii}), x_feat{ii});

        % write hdf5 list files for features
        file_id = fopen( fullfile(dir_dst, sprintf('feat_%s_%s.txt', ...
            feature_type_names{ii}, phase) ), 'w');
        fprintf(file_id, fullfile(dir_dst, sprintf('feat_%s_%s.h5\n', ...
            feature_type_names{ii}, phase) ) );
        fclose(file_id);
    end

    % save ground truth to hdf5(s)
    hdf5write( fullfile(dir_dst, sprintf('labels_%s.h5', phase) ), '/label', y );
    hdf5write( fullfile(dir_dst, sprintf('labels_scalar_%s.h5', phase) ), ...
        '/label', y_scalar );

    % write ground truth hdf5 list files (.txt)
    file_id = fopen( fullfile(dir_dst, sprintf('labels_%s.txt', phase) ), 'w');
    fprintf(file_id, fullfile(dir_dst, sprintf('labels_%s.h5\n', phase) ) );
    fclose(file_id);
    file_id = fopen( fullfile(dir_dst, sprintf('labels_scalar_%s.txt', phase) ), 'w');
    fprintf(file_id, fullfile(dir_dst, sprintf('labels_scalar_%s.h5', phase) ) );
    fclose(file_id);
end
