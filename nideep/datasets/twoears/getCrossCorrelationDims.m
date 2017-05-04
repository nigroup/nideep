function [num_freqChannels, num_lags] = getCrossCorrelationDims(featureNames)
% getCrossCorrelationDims identify dimensions of cross-correlation to reshape feature
% vector accordingly. 
%   [num_freqChannels, num_lags] = getCrossCorrelationDims(featureNames)
%   where argument featureNames describes first t, then frequencyBins, then
%   lag bins
%
%   See also twoears2Blob.

% get indicies of features with lag bins
is_feat_with_lag = cellfun(@(v) strncmpi(v, 'l', 1), featureNames, ...
    'un', false);
lag_idxs = find(cellfun(@(v) find(v), is_feat_with_lag));
featureNames_sub = featureNames(1, lag_idxs);
assert( isequal(length(lag_idxs), length(featureNames_sub) ), ...
    'Unexpected format for cross-correlation, found subset without lag bins');
clearvars is_feat_with_lag lag_idxs

% assume no. of freq. channels are kept constant throughout feat.
num_freqChannels = str2double( char( strrep( featureNames_sub{ 1 }( 2 ), '-ch', '' ) ) );

% get index of lag bins cross-correlation vector
% assume the index is fixed for all cross-correlation features
tmp = cellfun(@(v) strncmpi(v, 'l', 1), featureNames_sub{ 1 }, ...
    'un', false);
lag_idx = find( not(cellfun(@(v) isequal(v, 0), tmp)) ); % e.g. 11
lag_idxs_names = unique( cellfun(@(v) v(lag_idx), featureNames_sub) );
num_lags = length( cell2mat( cellfun(@(x) str2double(char(x(2:end))), ...
    lag_idxs_names, 'un', false) ) );


