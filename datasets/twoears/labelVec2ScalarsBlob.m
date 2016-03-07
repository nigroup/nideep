function [y_scalar] = labelVec2ScalarsBlob(y_vec)
% LABELVEC2SCALAR  reduce 4-D Blob matrix of 1-hot ground truth vector
%   to Blob with scalar ground truth.
%
%   A shape of (K x N) is expected for the ground truth y_vec where N is the number of samples
%   and K is the number of classes.
%   Sclarization only applies to 1 one-hot ground vectors
%
n = length(y_vec);
y_scalar = reshape(y_vec, [], n); % remove singleton dimensions

if any(sum(y_scalar, 1) > 1)
    warning('Cannot scalarize multi-labels. Skipping.')
    y_scalar = [];
else
    [class_ids, sample_idx] = find( y_scalar==1 );
    y_scalar = zeros( 1, n );
    y_scalar(:, sample_idx) = class_ids;
    y_scalar = y_scalar-1; % 1-based to zero-based class index
    y_scalar = reshape( y_scalar, 1, 1, 1, n ); % to 4-D Blob
end


