function [is_binaural] = isBinaural(featureNames)
% isBinaural  identify binaural features

is_combined = cellfun(@(v) strfind([v{:}], 'LRmean'), featureNames, ...
    'un', false);
is_combined = cellfun('isempty', is_combined);
is_mono = cellfun(@(v) strfind([v{:}], 'mono'), featureNames, 'un', false);
is_mono = cellfun('isempty', is_mono);
is_binaural = not(is_mono) & not(is_combined);
