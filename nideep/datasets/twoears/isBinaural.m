function [is_binaural] = isBinaural(featureNames)
% isBinaural  identify binaural features

is_not_combined = cellfun(@(v) strfind([v{:}], 'LRmean'), featureNames, ...
    'un', false);
is_not_combined = cellfun('isempty', is_not_combined);
is_not_mono = cellfun(@(v) strfind([v{:}], 'mono'), featureNames, 'un', false);
is_not_mono = cellfun('isempty', is_not_mono);
is_binaural = is_not_mono & is_not_combined;
