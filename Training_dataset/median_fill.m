
function depth_data_filled = median_fill(depth_data,filter_size,tab_i,tab_j)
%Fill GAPS with Median Filtering at places where there is 0
% ---- Inputs ----
% depth_data = depth image matrix
% filter_size = size of Median Filter
% [tab_i, tab_j] = index of dots where data has to be replaced 
% (ex : in case of filling zero values -> [tab_i,tab_j] = find(~image))

disp('Filling the gaps with Median Filtering...')
image = depth_data;

 if size(tab_i)==size(tab_j)
 else
     error('tab_i != tab_j')
 end

% Apply Median Filter
num_zeros = size(tab_i);
for ind = 1:num_zeros
    ii = tab_i(ind);
    jj = tab_j(ind);
    x_i = max(1,ii-filter_size);
    x_f = min(size(image,1), ii+filter_size);
    y_i = max(1, jj-filter_size);
    y_f = min(size(image,2), jj+filter_size);
    image(ii,jj) = median(image(x_i:x_f , y_i:y_f),'all');
end

depth_data_filled = image;

end