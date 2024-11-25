function z = zip_array(a,b)
% zip_array concatenates two arrays column wise.
%
% ## Comments
%
% If a and b are column vectors, they are concatenated row wise.
%
% ## Input Arguments
%
% `a` (_double_) - array
%
% `b` (_double_) - array
%
% ## Output Arguments
%
% `z` (_double_) - [column 1 of a, column 1 of b, column 2 of a, column 1 of b, ...]
%

if size(a,2) == 1 && size(b,2) == 1
    z = [zeros(size(a));zeros(size(a))];
    for idx = 1:length(a)
        z(2*idx-1) = a(idx);
        z(2*idx) = b(idx);
    end
else
    z = [zeros(size(a)),zeros(size(a))];
    for idx = 1:size(a,2)
        z(:,2*idx-1) = a(:,idx);
        z(:,2*idx) = b(:,idx);
    end
end

end