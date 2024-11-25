function out = maxabs(in)
% maxabs computes the maximum absolute value of an array.
%
% ## Comments
%
% _none_
%
% ## Input Arguments
%
% `in` (_double_) - array
%
% ## Output Arguments
%
% `out` (_double_) - maximum absolute value
%

out = max(abs(in(:)));

end