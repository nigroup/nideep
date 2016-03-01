function [] = twoears2hdf5(fpath, dir_dst)
% remove rows with zero entries in ground truth
load(fpath); % loads x, y, featureNames

dir_src = fileparts(fpath);
[~, phase] = fileparts(dir_src); % test or train from directory name
assert( strcmp(phase, 'test') | strcmp(phase, 'train'), 'Unable to determine phase (test vs. train).' );

[nozero, ~] = find( all( y~=0, 2 ) );
x2 = x( nozero, : );
y2 = y( nozero, : );
% exclude column for general class
y2( y2==-1 ) = 0;
general_col = find( strcmp( classnames, 'general' ) );
y2(:, general_col) = [];

% random shuffle
o = randperm( length( y2 ) );
x2 = x2( o, : );
y2 = y2( o, : );
% reshape to 4D caffe Blob
x2 = x2';
y2 = y2';

% assume first field contains feature name
feature_type_names = unique(cellfun(@(v) v(1), featureNames(1,:)));
x_feat = cell(size(feature_type_names));
for ii = 1 : numel(feature_type_names)
    disp(feature_type_names{ii})
    
    is_feat = cellfun(@(v) strfind([v{:}], feature_type_names{ii}), ...
        featureNames, 'un', false);
    feat_idxs = find(not(cellfun('isempty', is_feat)));
    
    t_idxs_names = unique(cellfun(@(v) v(4), featureNames(feat_idxs)));
    t_idxs = sort( cell2mat( cellfun(@(x) str2double(char(x(2:end))), t_idxs_names, 'un', false) ) );
    
    num_blocks = length( t_idxs );
    
    disp(min(t_idxs))
    disp(max(t_idxs))
    
    if strcmp(feature_type_names{ii}, 'amsFeatures')
        % T x F x mF x N
        % get indicies of features with modulation frequencies
        is_feat_with_mf = cellfun(@(v) strfind([v{:}], 'mf'), featureNames, 'un', false);
        mf_idxs = find(not(cellfun('isempty', is_feat_with_mf)));
        % assume no. of freq. channels are kept constant throughout feat.
        num_freqChannels = str2double( char( strrep( featureNames{ feat_idxs(1) }( 2 ), '-ch', '' ) ) );
        assert( isequal(mf_idxs, feat_idxs), ...
            'Unexpected format for amsFeatures, subset without modulation frequencies');
        % get index of modulation frequencies in ams feature vector
        % assume the index is fixed for all ams features
        tmp = cellfun(@(v) strfind(v, 'mf'), featureNames{ feat_idxs(1) }, ...
            'un', false);
        mf_idx = find( not(cellfun('isempty', tmp)) ); % e.g. 8
        mf_idxs_names = unique( cellfun(@(v) v(mf_idx), featureNames(1, feat_idxs)) );
        num_mod_freq = length( cell2mat( cellfun(@(x) str2double(char(x(3:end))), ...
            mf_idxs_names, 'un', false) ) );
        % first t, then f, then mf
    elseif strcmp(feature_type_names{ii}, 'ratemap')
        %  T x F x 1 x N
        num_freqChannels = str2double( char( strrep( featureNames{1}( 2 ), '-ch', '' ) ) );
        num_mod_freq = 1;
    else
        error('feature %s type not supported', feature_type_names{ii});
    end
    x_feat_tmp = x2(feat_idxs, :);
    x_feat_tmp = reshape( x_feat_tmp, num_blocks, num_freqChannels, num_mod_freq, length( x_feat_tmp ) );
    x_feat{ii} = x_feat_tmp;
end % format features

% reshape multi-label ground truth vectors to 4-D Blob
y2 = reshape( y2, 1, length( classnames ) - 1, 1, length( y2 ) );

% labels in one-hot format for softmax loss
ys = reshape(y2, length( classnames ) - 1, length(y2));
[class_ids, sample_idx] = find(ys==1);
ys2 = zeros(1, length(ys));
ys2(:, sample_idx) = class_ids;
ys2 = ys2-1;
% to 4-D Blob
ys2 = reshape(ys2, 1, 1, 1, length(y2));

for ii = 1 : numel(feature_type_names)
    % save formatted features to file
    fname = sprintf('feat_%s_%s.h5', feature_type_names{ii}, phase);
    hdf5write( fullfile(dir_dst, fname), strcat('/', feature_type_names{ii}), x_feat{ii});
    
    % write hdf5 list files for features
    file_id = fopen( fullfile(dir_dst, sprintf('feat_%s_%s.txt', feature_type_names{ii}, phase) ),'w');
    fprintf(file_id, fullfile(dir_dst, sprintf('feat_%s_%s.h5\n', feature_type_names{ii}, phase) ) );
    fclose(file_id);
end

% save ground truth to file(s)
hdf5write( fullfile(dir_dst, sprintf('labels_%s.h5', phase) ), '/label', y2 );
hdf5write( fullfile(dir_dst, sprintf('labels_1hot_%s.h5', phase) ), '/label', ys2 );

% write hdf5 list files
file_id = fopen( fullfile(dir_dst, sprintf('labels_%s.txt', phase) ),'w');
fprintf(file_id, fullfile(dir_dst, sprintf('labels_%s.h5\n', phase) ) );
fclose(file_id);
file_id = fopen( fullfile(dir_dst, sprintf('labels_1hot_%s.txt', phase) ),'w');
fprintf(file_id, fullfile(dir_dst, sprintf('labels_1hot_%s.h5', phase) ) );
fclose(file_id);
