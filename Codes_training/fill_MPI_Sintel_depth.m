clear all;
close all;
%%%%% PARAMETERS TO DEFINE ------------------------------------------------
rgb = 1;
% ------------------------------------------------ ------------------------
disp('Processing MPI Single Depth Dataset...')
D = '/Users/aliceruget/Documents/PhD/Dataset/MPI_Sintel_Depth/MPI-Sintel-stereo-training-20150305/training';
saveD = '/Users/aliceruget/Documents/PhD/Dataset/Processed_MPI_Sintel_Depth_RGB/';
S = dir(fullfile(D,'*'));
depth_folder = dir(fullfile(D,'disparities_viz'));
intensity_folder = dir(fullfile(D,'clean_left'));
N_depth = setdiff({depth_folder([depth_folder.isdir]).name},{'.','..','zip','non2-6'}); 
N_intensity = setdiff({intensity_folder([intensity_folder.isdir]).name},{'.','..','zip','non2-6'}); % list of subfolders of D.

depth_data_MPI = cell(1,23);
intensity_data_MPI = cell(1,23);
%depth_data_MPI = {};
%intensity_data_MPI = {}; 
tab_filter_size_intensity_MPI = zeros(23,1);
tab_filter_size_depth_MPI = zeros(23,1);
tab_nb_inf_depth = zeros(23,1);
tab_nb_nan_depth = zeros(23,1);
tab_nb_zeros_depth = zeros(23,1);
tab_nb_inf_intensity = zeros(23,1);
tab_nb_nan_intensity = zeros(23,1);

image_index = 1;

for image = 1:numel(N_depth) % loop on image in year
    disp(N_depth{image})
    if ~exist(fullfile(saveD,N_depth{image}), 'dir')
        mkdir(fullfile(saveD,N_depth{image}))
    end

    % ---------------------------------------------------------------------
    % -- INIT -- 
    % ---------------------------------------------------------------------
    depth_image = imread(fullfile(D,'disparities_viz',N_depth{image},'frame_0001.png'));% improve by specifying the file extension.
    if rgb ==1
        intensity_image = imread(fullfile(D,'clean_left',N_intensity{image},'frame_0001.png'));% improve by specifying the file extension.
    else
        intensity_image = rgb2gray(imread(fullfile(D,'clean_left',N_intensity{image},'frame_0001.png')));% improve by specifying the file extension.
    end
        
    depth_image = im2double(depth_image);
    intensity_image = im2double(intensity_image);
    if rgb == 0
        figure;imagesc(depth_image);colormap(jet);colorbar
        saveas(gcf, fullfile(saveD, N_depth{image}, 'depth_init.png'))
        figure;imagesc(intensity_image);colormap(gray);colorbar
        saveas(gcf, fullfile(saveD, N_depth{image}, 'intensity_init.png'))
        figure; DepthImage = depth_image;r1 = size(DepthImage,1);c1 = size(DepthImage,2);
        DProc=reshape(DepthImage,r1,c1);
        scatter3(kron(ones(r1,1),(1:c1)'),reshape(DProc(end:-1:1,:)',r1*c1,1)',kron((1:r1)',ones(c1,1)),80,reshape(DProc(end:-1:1,:)',r1*c1,1)','.'),
        colorbar;colormap(jet);view(-169.5624,8.3685);
        saveas(gcf, fullfile(saveD, N_depth{image}, 'depth_init_3D.png'))
    end
    

    % ---------------------------------------------------------------------
    % -- DEPTH MEDIAN FILTERING FOR OUT-OF-VALUES PIXELS -- 
    % For depth in MPI : out-of-values = inf, nan
    % ---------------------------------------------------------------------
    filter_size_depth = 1;
    
    % save placements of out-of-values pixels
    matrice_1 = zeros(size(depth_image));
    [tab_i_1, tab_j_1] =  find(isnan(depth_image)); 
    for t = 1:size(tab_i_1)
        matrice_1(tab_i_1(t), tab_j_1(t)) = 1;
    end
    tab_nb_nan_depth(image_index) = sum(matrice_1(:));
%     figure; imagesc(matrice_1);title('Nan pixels of depth image');colorbar
%     saveas(gcf, fullfile(saveD, N_depth{image}, 'nan_pix_depth.png'))
    
    matrice_1 = zeros(size(depth_image));
    [tab_i_1, tab_j_1] =  find(isinf(depth_image)); 
    for t = 1:size(tab_i_1)
        matrice_1(tab_i_1(t), tab_j_1(t)) = 1;
    end
    tab_nb_inf_depth(image_index) = sum(matrice_1(:));
%     figure; imagesc(matrice_1);title('Inf pixels of depth image');colorbar
%     saveas(gcf, fullfile(saveD, N_depth{image}, 'inf_pix_depth.png'))
    
    while filter_size_depth < size(depth_image, 1)/30
        [tab_i, tab_j] =  find(isnan(depth_image) | isinf(depth_image));
        
        depth_image_new = median_fill(depth_image, filter_size_depth, tab_i, tab_j);
        % stop before if all values already deleted
        [tab_test,tab_test] = find(isnan(depth_image_new) | isinf(depth_image_new));
        if isempty(tab_test)
            break
        end
        filter_size_depth = filter_size_depth +1;
    end
    tab_filter_size_depth_MPI(image_index) = filter_size_depth;
    depth_data_MPI{1, image_index} = depth_image_new;
    %depth_data_MPI = cat(2,depth_data_MPI,depth_image_new);
    
    %figure;imagesc(depth_image_new);colorbar;colormap(jet)
%      
    % ---------------------------------------------------------------------
    % -- DEPTH MEDIAN FILTERING FOR OUT-OF-VALUES PIXELS --
    %Particular cases : alley_1, market_6, sleeping_2
    % ---------------------------------------------------------------------
    matrice_1 = zeros(size(depth_image));
    [tab_i_1, tab_j_1] =  find(~(depth_image)); 
    for t = 1:size(tab_i_1)
        matrice_1(tab_i_1(t), tab_j_1(t)) = 1;
    end
%     figure; imagesc(matrice_1);title('Zero pixels of depth image');colorbar
%     saveas(gcf, fullfile(saveD, N_depth{image}, 'zero_pix_depth.png'))
    tab_nb_zeros_depth(image_index) = sum(matrice_1(:));
    
    if strcmp(N_depth{image},'alley_1') || strcmp(N_depth{image},'market_6') || strcmp(N_depth{image},'sleeping_2')
        disp('Zeros are deleted')
        filter_size_depth = 1;
        while filter_size_depth < size(depth_image, 1)/30
            [tab_i, tab_j] =  find(~(depth_image));
        
            depth_image_new = median_fill(depth_image, filter_size_depth, tab_i, tab_j);
            % stop before if all values already deleted
            [tab_test,tab_test] = find(~(depth_image_new));
            if isempty(tab_test)
                break
            end
            filter_size_depth = filter_size_depth +1;
        end
        figure;imagesc(depth_image_new);colorbar;colormap(jet)
        depth_data_MPI{1, image_index} = depth_image_new;
%         saveas(gcf, fullfile(saveD, N_depth{image}, 'depth_post.png'))
    
%         figure;DepthImage = depth_image_new;r1 = size(DepthImage,1);c1 = size(DepthImage,2);
%         DProc=reshape(DepthImage,r1,c1);
%         scatter3(kron(ones(r1,1),(1:c1)'),reshape(DProc(end:-1:1,:)',r1*c1,1)',kron((1:r1)',ones(c1,1)),80,reshape(DProc(end:-1:1,:)',r1*c1,1)','.'),
%         colorbar;colormap(jet);view(-169.5624,8.3685)
%         saveas(gcf, fullfile(saveD, N_depth{image}, 'depth_post_3D.png'))
        
    else
        disp('Zeros are not deleted')
        
%         saveas(gcf, fullfile(saveD, N_depth{image}, 'depth_post.png'))
%     
%         figure;DepthImage = depth_image_new;r1 = size(DepthImage,1);c1 = size(DepthImage,2);
%         DProc=reshape(DepthImage,r1,c1);
%         scatter3(kron(ones(r1,1),(1:c1)'),reshape(DProc(end:-1:1,:)',r1*c1,1)',kron((1:r1)',ones(c1,1)),80,reshape(DProc(end:-1:1,:)',r1*c1,1)','.'),
%         colorbar;colormap(jet);view(-169.5624,8.3685)
%         saveas(gcf, fullfile(saveD, N_depth{image}, 'depth_post_3D.png')) 
        
    end
    % ---------------------------------------------------------------------
    % -- INTENSITY MEDIAN FILTERING FOR OUT-OF-VALUES PIXELS --
    %For intensity in MPI : out-of-values = inf, nan
    % ---------------------------------------------------------------------
    if rgb == 0
        filter_size_intensity = 1;
        % save placements of out-of-values pixels
        matrice = zeros(size(intensity_image));
        [tab_i, tab_j] =  find(isnan(intensity_image));
        for t = 1:size(tab_i)
            matrice(tab_i(t), tab_j(t)) = 1;
         end
%         figure; imagesc(matrice);title('Nan pixels of intensity image');colorbar
%         saveas(gcf, fullfile(saveD, N_depth{image}, 'nan_pix_intensity.png'))
        tab_nb_nan_intensity(image_index) = sum(matrice(:));
        
        matrice = zeros(size(intensity_image));
        [tab_i, tab_j] =  find( isinf(depth_image));
        for t = 1:size(tab_i)
            matrice(tab_i(t), tab_j(t)) = 1;
        end
        tab_nb_inf_intensity(image_index) = sum(matrice(:));
%         figure; imagesc(matrice);title('Inf pixels of intensity image');colorbar
%         saveas(gcf, fullfile(saveD, N_depth{image}, 'inf_pix_intensity.png'))
%         
        
        while filter_size_intensity < size(intensity_image, 1)/30
            [tab_i, tab_j] =  find(isnan(intensity_image) | isinf(intensity_image));
            
            if isempty(tab_i)
                filter_size_intensity = 0;
                intensity_image_new = intensity_image;
                break
            end
            intensity_image_new = median_fill(intensity_image, filter_size_intensity, tab_i, tab_j);
            % stop before if all values already deleted
            [tab_test,tab_test] = find(isnan(intensity_image_new) | isinf(intensity_image_new));
            if isempty(tab_test)
                break
            end
            filter_size_intensity = filter_size_intensity + 1;
        end
        tab_filter_size_intensity_MPI(image_index) = filter_size_intensity;
        %intensity_data_MPI = cat(2,intensity_data_MPI,intensity_image_new);
        intensity_data_MPI{1, image_index} = intensity_image_new;
        figure;imagesc(intensity_image_new);colorbar;colormap(gray)
        saveas(gcf, fullfile(saveD, N_depth{image}, 'intensity_post.png'))
        
        image_index = image_index + 1;
    else 
        
        filter_size_intensity = 1;
        intensity_image_new = zeros(size(intensity_image));
        for c = 1:3
            intensity_image_c = squeeze(intensity_image(:,:,c));
            while filter_size_intensity < size(intensity_image_c, 1)/30
                [tab_i, tab_j] =  find(isnan(intensity_image_c) | isinf(intensity_image_c));
                
                if isempty(tab_i)
                    filter_size_intensity = 0;
                    intensity_image_new_c = intensity_image_c;
                    break
                end
                intensity_image_new_c = median_fill(intensity_image_c, filter_size_intensity, tab_i, tab_j);
                % stop before if all values already deleted
                [tab_test,tab_test] = find(isnan(intensity_image_new_c) | isinf(intensity_image_new_c));
                if isempty(tab_test)
                    break
                end
                filter_size_intensity = filter_size_intensity + 1;
            end
            tab_filter_size_intensity_MPI(image_index) = filter_size_intensity;
            intensity_image_new(:,:,c) = intensity_image_new_c;
            figure
            imagesc(squeeze(intensity_image_new(:,:,c)))
            colorbar
        end
        %intensity_data_MPI = cat(2,intensity_data_MPI,intensity_image_new);
        intensity_data_MPI{1, image_index} = intensity_image_new;
        
        
        image_index = image_index + 1;
    end
        
        
        

end

save(fullfile(saveD, 'depth_data.mat'),'depth_data_MPI');
save(fullfile(saveD, 'intensity_data.mat'),'intensity_data_MPI');
save(fullfile(saveD,N_depth{image}, 'tab_filter_size_intensity.mat'),'tab_filter_size_intensity_MPI');
save(fullfile(saveD,N_depth{image}, 'tab_filter_size_depth.mat'),'tab_filter_size_depth_MPI');
save(fullfile(saveD,N_depth{image}, 'tab_nb_inf_intensity.mat'),'tab_nb_inf_intensity');
save(fullfile(saveD,N_depth{image}, 'tab_nb_nan_intensity.mat'),'tab_nb_nan_intensity');
save(fullfile(saveD,N_depth{image}, 'tab_nb_inf_depth.mat'),'tab_nb_inf_depth');
save(fullfile(saveD,N_depth{image}, 'tab_nb_nan_depth.mat'),'tab_nb_nan_depth');
save(fullfile(saveD,N_depth{image}, 'tab_nb_zeros_depth.mat'),'tab_nb_zeros_depth');
disp('Done')
