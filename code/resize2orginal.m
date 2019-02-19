function resized = resize2orginal(test_predlbl, test_t1_org)
    [row, col] = size(test_t1_org);
    resized = ones(row, col, 'double');
    for i = 1:row
        for j = 1:col
            test_predlbl = double(test_predlbl);
            resized(i,j) = test_predlbl(i,j);
        end
    end

end