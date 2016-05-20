function [x_feat, feature_type_names, y] = twoears2Blob(x, featureNames, y)
% twoears2Blob  reshape feature and ground truth vectors into 4-D Blob for caffe
%   For the feature vector x it expects a shape of (N x D)
%   where N is the number of samples and D is the total no. of features
%
%   For the ground truth vectors y it expects a shape of (N x K)
%   where N is the number of samples and K is the number of classes.
%   The ground truth vectors can be one-hot or multi-label vectors
%
%   See also twoears2hdf5.
x = x';
y = y';

% assume first field contains feature name
feature_type_names = unique( cellfun(@(v) v(1), featureNames(1,:)) );
x_feat = cell( size(feature_type_names) );
for ii = 1 : numel(feature_type_names)
    disp(feature_type_names{ii})
    
    % Determine time bins in a single block.
    % We assume the block size is constant within a feature type
    is_feat = cellfun(@(v) strfind([v{:}], feature_type_names{ii}), ...
        featureNames, 'un', false);
    feat_idxs = find(not(cellfun('isempty', is_feat)));
    
    t_idxs_names = unique(cellfun(@(v) v(4), featureNames(feat_idxs)));
    t_idxs = sort( cell2mat( cellfun(@(x) str2double(char(x(2:end))), ...
        t_idxs_names, 'un', false) ) );
    
    num_blocks = length( t_idxs );
    
    disp([min(t_idxs), max(t_idxs)]);
    
    if strcmp(feature_type_names{ii}, 'amsFeatures')
        % T x F x mF x N
        [num_freqChannels, num_mod] = getAMSFeaturesDims(featureNames(feat_idxs));
    elseif strcmp(feature_type_names{ii}, 'ratemap')
        %  T x F x 1 x N
        num_freqChannels = str2double( char( strrep( featureNames{feat_idxs(1)}( 2 ), '-ch', '' ) ) );
        num_mod = 1;
    elseif strcmp(feature_type_names{ii}, 'crosscorrelation')
        %  T x F x nLags x N
        [num_freqChannels, num_mod] = getCrossCorrelationDims(featureNames(feat_idxs));
    elseif strcmp(feature_type_names{ii}, 'ild')
        %  T x F x 1 x N
        num_freqChannels = str2double( char( strrep( featureNames{feat_idxs(1)}( 2 ), '-ch', '' ) ) );
        num_mod = 1;
    else
        warning('Skipping unsupported feature type %s.', feature_type_names{ii});
    end
    x_feat_tmp = x(feat_idxs, :);
    x_feat{ii} = reshape( x_feat_tmp, ...
        num_blocks, num_freqChannels, num_mod, ...
        size( x_feat_tmp, 2 ) );
    clearvars x_feat_tmp;
    x(feat_idxs, :) = []; % we don't need those features anymore
    featureNames(feat_idxs) = [];
end % format features

% reshape multi-label ground truth vectors to 4-D Blob
y = reshape( y, 1, [], 1, length( y ) );

