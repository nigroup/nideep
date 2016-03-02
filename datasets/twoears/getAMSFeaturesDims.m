function [num_freqChannels, num_mod_freq] = getAMSFeaturesDims(featureNames)
% getAMSFeaturesDims identify dimensions of amsFeatures to reshape feature
% vector accordingly. 
%   [num_freqChannels, num_mod_freq] = getAMSFeaturesDims(featureNames)
%   where argument featureNames describes first t, then frequencyBins, then
%   modulation frequencies
%
%   See also twoears2Blob.

% get indicies of features with modulation frequencies
is_feat_with_mf = cellfun(@(v) strfind([v{:}], 'mf'), featureNames, 'un', false);
mf_idxs = find(not(cellfun('isempty', is_feat_with_mf)));
featureNames_sub = featureNames(1, mf_idxs);

% assume no. of freq. channels are kept constant throughout feat.
num_freqChannels = str2double( char( strrep( featureNames_sub{ 1 }( 2 ), '-ch', '' ) ) );
assert( isequal(length(mf_idxs), length(featureNames_sub) ), ...
    'Unexpected format for amsFeatures, subset without modulation frequencies');
% get index of modulation frequencies in ams feature vector
% assume the index is fixed for all ams features
tmp = cellfun(@(v) strfind(v, 'mf'), featureNames_sub{ 1 }, ...
    'un', false);
mf_idx = find( not(cellfun('isempty', tmp)) ); % e.g. 8
mf_idxs_names = unique( cellfun(@(v) v(mf_idx), featureNames_sub) );
num_mod_freq = length( cell2mat( cellfun(@(x) str2double(char(x(3:end))), ...
    mf_idxs_names, 'un', false) ) );


