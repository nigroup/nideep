function [y_vec] = removeClassFromVec(y_vec, class_col)
% REMOVECLASSFROMVEC  remove a class colum from ground truth vectors
%
n = length(y_vec);
classes_dim = find(and(size(y_vec) > 1, size(y_vec) ~= n));
% determine where the class dimension in the 4D blob is.
if classes_dim == 1
    y_vec(class_col, :, :, :, :, :) = [];
elseif classes_dim == 2
    y_vec(:, class_col, :, :, :, :) = [];
elseif classes_dim == 3
    y_vec(:, :, class_col, :, :, :) = [];
else
    error('Unexpected ground truth array size.');
end


