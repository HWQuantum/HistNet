%PLOT RESULTS
%% Robustness noise 
close all
map_nonoise = importdata('/Users/aliceruget/Documents/PhD/HistSR_Net_AR/Results/noisy_i3000_SBR0_4_MPI_l1_loss/Robustness_SBR_i/rmse_map_2d_plot_i3000_SBR_0_4_l1.mat');
map_noisy = importdata('/Users/aliceruget/Documents/PhD/HistSR_Net_AR/Results/noisy_MPI_i10_SBR0_004_l1/robustness_dif_sbr/rmse_map_2d_plot_i10_sbr0004.mat');
subplot(1,2,1)
imagesc(map_noisy)
hold on 
plot(10,8.5, 'r', 'Marker', '*', 'Linewidth', 2,'MarkerSize',12,'LineStyle', 'none')
caxis([0 0.4])
colorbar
axis image

xticks(linspace(1,28,8))
xticklabels([5*10^(-5), 0.0005, 0.005, 0.05, 0.5, 5, 50])
yticks(linspace(1,28,8))
yticklabels({0.04 0.4 4 40 400 4000 40000})
legend('noise level of training dataset')
xlabel('SBR','interpreter', 'latex','fontsize',13)
ylabel('ppp','interpreter', 'latex','fontsize',13)
title('(a)', 'interpreter', 'latex','fontsize',13)
subplot(1,2,2)
imagesc(map_nonoise)
hold on 
plot(19.5,19, 'r', 'Marker', '*','Linewidth', 2,'MarkerSize',12,'LineStyle', 'none')
caxis([0 0.4])
colorbar
axis image
xticks([0 4 8 12 16 20 24])
xticklabels([5*10^(-5), 0.0005, 0.005, 0.05, 0.5, 5, 50])
yticks([0 4 8 12 16 20 24])
yticklabels({0.04 0.4 4 40 400 4000 40000})
legend('noise level of training dataset')
title('(b)', 'interpreter', 'latex','interpreter', 'latex','fontsize',13)
xlabel('SBR','interpreter', 'latex','fontsize',13)
ylabel('ppp','interpreter', 'latex','fontsize',13)
set(gcf, 'Position', [ 273         423        1146         375])
%% Plot Simulated without 3D
close all
%PathName = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Simulate_data/*SBR0.02';
PathName = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Simulate_data/*SBR2';
Path = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Simulate_data/'
d = dir(fullfile(PathName));
d = d([d.isdir] & ~ismember({d.name},{'.','..'}));
linewidth = 3;
ecart_ligne = 0.001
ecart_row = 0.125;
n = 1;
for n = 1:numel(d)
%if n == 1
    if n == 1
        coord1 = 200:500;
        coord2 = 600:800;
    else 
        coord1 = 200:400;
        coord2 = 200:400;
    end
 
    d2 = fullfile(Path, d(n).name);
    disp(d2)
    
    ax1 = subplot(numel(d)*2,7,(n-1)*14+1);
    i = 1
    %set(ax1, 'Position', [(i-1)*0.125+(i)*ecart_ligne 0.2+0.156 0.1250 0.1250]);
    df = importdata(fullfile(d2,'data', '0_Df.mat'));
    imagesc(df)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax1,parula)
    save(fullfile(d2,'Results_depth', 'Label.mat'), 'df')
    if n==1
        title({'\textbf{Ground Truth Depth}','(a)'},'interpreter','latex','linewidth',1)
    else
        title('(h)', 'interpreter','latex')
    end
    
    
    ax1_bis = subplot(numel(d)*2,7,(n-1)*14+8);
    df_small = df(coord1, coord2);
    imagesc(df_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax1_bis,parula)
    if n==1
        title('(a*)','interpreter','latex','linewidth',linewidth)
    else
        title('(h*)', 'interpreter','latex')
    end
    i = 1
    %set(ax1_bis, 'Position', [(i-1)*0.125+(i)*ecart_ligne 0.156 0.1250 0.1250]);
    
    
    ax2 = subplot(numel(d)*2,7,((n-1)*14+2));
    inten = imread(fullfile(d2,'data', '0_RGB.bmp'));
    imagesc(inten)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    %caxis([0,1])
    colormap(ax2,gray)
    if n == 1
        title({'\textbf{Ground Truth Intensity}','(b)'},'interpreter','latex')
    else
        title('(i)','interpreter','latex')
    end
    i = 2
    %set(ax2, 'Position', [(i-1)*0.125+(i)*0.0110 0.156+0.2 0.1250 0.1250]);
    
    ax2_bis = subplot(numel(d)*2,7,(n-1)*14+9);
    inten_small = inten(coord1, coord2);
    imagesc(inten_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    colormap(ax2_bis,gray)
    if n==1
        title('(b*)','interpreter','latex','linewidth',linewidth)
    else
        title('(i*)', 'interpreter','latex')
    end
    i = 2
    %set(ax2_bis, 'Position', [(i-1)*0.125+(i)*0.0110 0.156 0.1250 0.1250]);
    
    ax3 = subplot(numel(d)*2,7,(n-1)*14+3);
    interp_image = importdata(fullfile(d2,'data', '0_Df_down.mat'));
    imagesc(interp_image)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    caxis([0,1])
    colormap(ax3,parula)
    rmse_val = cal_rmse(interp_image,df);
    ae_val = cal_ae(interp_image,df);
    disp(strcat('NNI', 'rmse=', num2str(rmse_val),'_ae=',num2str(ae_val)))
    save(fullfile(d2,'Results_depth', 'result_nni.mat'), 'interp_image')
    %txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    txt = strcat('ADE=',sprintf('%.4f',ae_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n==1
        title({'\textbf{Nearest Interpolation}','(c)'},'interpreter','latex','linewidth',linewidth)
    else
        title('(j)', 'interpreter','latex')
    end  
    i = 3
    %set(ax3, 'Position', [(i-1)*0.125+(i)*0.0110 0.156+0.2 0.1250 0.1250]); 
    
    
    ax3_bis = subplot(numel(d)*2,7,(n-1)*14+10);
    interp_image_small = interp_image(coord1, coord2);
    imagesc(interp_image_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax3_bis,parula)
    if n==1
        title('(c*)','interpreter','latex','linewidth',9)
    else
        title('(j*)', 'interpreter','latex')
    end
    i = 3
    %set(ax3_bis, 'Position', [(i-1)*0.125+(i)*0.0110 0.156 0.1250 0.1250]);
    
    
    ax4 = subplot(numel(d)*2,7,(n-1)*14+4);
    tic
    B = imguidedfilter(interp_image,inten);
    timeElapsed = toc
    disp(strcat('Guided Filtering=', num2str(timeElapsed)))
    imagesc(B)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    caxis([0,1])
    colormap(ax4,parula)
    rmse_val = cal_rmse(B,df);
    ae_val = cal_ae(B,df);
    disp(strcat('Guided Filtering', 'rmse=', num2str(rmse_val),'_ae=',num2str(ae_val)))
    save(fullfile(d2,'Results_depth', 'Result_guided_filtering.mat'), 'B')
    %txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    txt = strcat('ADE=',sprintf('%.4f',ae_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n==1
        title({'\textbf{Guided Image Filtering}','(d)'},'interpreter','latex')
    else
        title('(k)', 'interpreter','latex')
    end
    i = 4
    %set(ax4, 'Position', [(i-1)*0.125+(i)*0.0110 0.156+0.2 0.1250 0.1250]); 
    


    ax4_bis = subplot(numel(d)*2,7,(n-1)*14+11);
    B_small = B(coord1, coord2);
    imagesc(B_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax4_bis,parula)
    if n==1
        title('(d*)','interpreter','latex','linewidth',9)
    else
        title('(k*)', 'interpreter','latex')
    end
    i = 4
    %set(ax4_bis, 'Position', [(i-1)*0.125+(i)*0.0110 0.156 0.1250 0.1250]);
    
    ax5 = subplot(numel(d)*2,7,(n-1)*14+5);
    sol = importdata(fullfile(d2,'Results_depth', 'parameters.mat'));
    sol = sol.result;
    sol = squeeze(sol);
    save(fullfile(d2,'Results_depth', 'result_DepthSR_Net.mat'), 'sol')
    
    imagesc(sol)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    caxis([0,1])
    colormap(ax5,parula)
    rmse_val = cal_rmse(sol, df);
    ae_val = cal_ae(sol,df);
    disp(strcat('DepthNet', 'rmse=', num2str(rmse_val),'_ae=',num2str(ae_val)))
    
    %txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    txt = strcat('ADE=',sprintf('%.4f',ae_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n==1
        %title(ax5,{'\textbf{DepthSR-Net}','(e)'},'interpreter','latex')
        title(ax5,{'\textbf{Guo \textit{et al.} 2019}','(e)'},'interpreter','latex')
        
    else
        title('(l)', 'interpreter','latex')
    end
    i = 5
    %set(ax5, 'Position', [(i-1)*0.125+(i)*0.0110 0.156+0.2 0.1250 0.1250]); 
    

    ax5_bis = subplot(numel(d)*2,7,(n-1)*14+12);
    sol_small = sol(coord1, coord2);
    imagesc(sol_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax5_bis,parula)
    if n==1
        title('(e*)','interpreter','latex','linewidth',9)
    else
        title('(l*)', 'interpreter','latex')
    end
    i = 5
    %set(ax5_bis, 'Position', [(i-1)*0.125+(i)*0.0110 0.156 0.1250 0.1250]);
    
    
    ax6 = subplot(numel(d)*2,7,(n-1)*14+6);
    %sol = importdata(fullfile(d2,'Results/Reconstructions', '0recon.mat'));
    if n == 1
        %sol = importdata('/Users/aliceruget/Documents/PhD/Optica/data_LAB_JONATHAN_ALICE_TEST_FUSION_SYNTH.mat');
        sol = importdata('/Users/aliceruget/Documents/PhD/Optica/OneDrive_1_17-09-2020/data_LAB_JONATHAN_ALICE_TEST_FUSION_SYNTH_ART.mat')
        sol = squeeze(sol.D_processed(:,:,2)+0.035);
    else 
        sol = importdata('/Users/aliceruget/Documents/PhD/Optica/OneDrive_1_17-09-2020/data_LAB_JONATHAN_ALICE_TEST_FUSION_SYNTH_MASK.mat')
        sol = squeeze(sol.D_processed(:,:,2)+0.035);
    end
    imagesc(sol)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    caxis([0,1])
    colormap(ax6,parula)
    rmse_val = cal_rmse(sol, df);
    
    ae_val = cal_ae(sol,df);
    disp(strcat('Optica', 'rmse=', num2str(rmse_val),'_ae=',num2str(ae_val)))
    
    %txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    txt = strcat('ADE=',sprintf('%.4f',ae_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n==1
        title(ax6,{'\textbf{Gyongy \textit{et al.} 2020}','(f)'},'interpreter','latex')
    else
        title('(m)', 'interpreter','latex')
    end
    i = 6
    %set(ax6, 'Position', [(i-1)*0.125+(i)*0.0110 0.156+0.2 0.1250 0.1250]); 
    
    ax6_bis = subplot(numel(d)*2,7,(n-1)*14+13);
    sol_small = sol(coord1, coord2);
    imagesc(sol_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax6_bis,parula)
    if n==1
        title('(f*)','interpreter','latex','linewidth',9)
    else
        title('(m*)', 'interpreter','latex')
    end
    i = 6
    %set(ax6_bis, 'Position', [(i-1)*0.125+(i)*0.0110 0.156 0.1250 0.1250]);
    
    
    ax7 = subplot(numel(d)*2,7,(n-1)*14+7);
    sol = importdata(fullfile(d2,'Results/Reconstructions', '0recon.mat'));
    imagesc(sol)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    caxis([0,1])
    colormap(ax7,parula)
    rmse_val = cal_rmse(sol, df);
    ae_val = cal_ae(sol,df);
    disp(strcat('Hist', 'rmse=', num2str(rmse_val),'_ae=',num2str(ae_val)))
    save(fullfile(d2,'Results_depth', 'result_HistNet.mat'), 'interp_image')
    %txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    txt = strcat('ADE=',sprintf('%.4f',ae_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n==1
        title(ax7,{'\textbf{Proposed HistNet}','(g)'},'interpreter','latex')
    else
        title('(n)', 'interpreter','latex')
    end
    i = 7
    %set(ax7, 'Position', [(i-1)*0.125+(i)*0.0110 0.156+0.2 0.1250 0.1250]); 
    

    ax7_bis = subplot(numel(d)*2,7,(n-1)*14+14);
    sol_small = sol(coord1, coord2);
    imagesc(sol_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax7_bis,parula)
    if n==1
        title('(g*)','interpreter','latex','linewidth',9)
    else
        title('(n*)', 'interpreter','latex')
    end
    i = 7
    %set(ax7_bis, 'Position', [(i-1)*0.125+(i)*0.0110 0.156 0.1250 0.1250]);
    
end

set(gcf, 'Position',[118          82        1249         723])
%% Plot Simulated without Optica
close all
PathName = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Simulate_data/*SBR2';
Path = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Simulate_data/'
d = dir(fullfile(PathName));
d = d([d.isdir] & ~ismember({d.name},{'.','..'}));
 
 
for n = 1:numel(d)
    if n == 1
        coord1 = 200:500;
        coord2 = 600:800;
    else 
        coord1 = 200:400;
        coord2 = 200:400;
    end
 
    d2 = fullfile(Path, d(n).name);
    disp(d2)
    
    ax1 = subplot(numel(d)*2,6,(n-1)*12+1);
    df = importdata(fullfile(d2,'data', '0_Df.mat'));
    imagesc(df)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax1,parula)
    if n==1
        title({'\textbf{Ground Truth Depth}','(a)'},'interpreter','latex','linewidth',9)
    else
        title('(g)', 'interpreter','latex')
    end
    
    ax1bis = subplot(numel(d)*2,6,(n-1)*12+7)
    df_small = df(coord1, coord2);
    imagesc(df_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax1bis,parula)
    if n==1
        title('(a*)','interpreter','latex','linewidth',9)
    else
        title('(g*)', 'interpreter','latex')
    end
    
    ax2 = subplot(numel(d)*2,6,((n-1)*12+2))
    inten = imread(fullfile(d2,'data', '0_RGB.bmp'));
    imagesc(inten)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    colormap(ax2,gray)
    if n == 1
        title({'\textbf{Ground Truth Intensity}','(b)'},'interpreter','latex')
    else
        title('(h)','interpreter','latex')
    end
    
    ax2_bis = subplot(numel(d)*2,6,(n-1)*12+8)
    inten_small = inten(coord1, coord2);
    imagesc(inten_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    colormap(ax2_bis,gray)
    if n==1
        title('(b*)','interpreter','latex','linewidth',9)
    else
        title('(h*)', 'interpreter','latex')
    end
    
    ax3 = subplot(numel(d)*2,6,(n-1)*12+3);
    interp_image = importdata(fullfile(d2,'data', '0_Df_down.mat'));
    imagesc(interp_image)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    caxis([0,1])
    colormap(ax3,parula)
    rmse_val = cal_rmse(interp_image,df);
    txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n==1
        title({'\textbf{Nearest Interpolation}','(c)'},'interpreter','latex')
    else
        title('(i)', 'interpreter','latex')
    end  
      
    
    
    ax3_bis = subplot(numel(d)*2,6,(n-1)*12+9)
    interp_image_small = interp_image(coord1, coord2);
    imagesc(interp_image_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax3_bis,parula)
    if n==1
        title('(c*)','interpreter','latex','linewidth',9)
    else
        title('(i*)', 'interpreter','latex')
    end
    
    ax4 = subplot(numel(d)*2,6,(n-1)*12+4)
    B = imguidedfilter(interp_image,inten);
    imagesc(B)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    caxis([0,1])
    colormap(ax4,parula)
    rmse_val = cal_rmse(B,df);
    txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n==1
        title({'\textbf{Guided Filtering}','(d)'},'interpreter','latex')
    else
        title('(j)', 'interpreter','latex')
    end


    ax4_bis = subplot(numel(d)*2,6,(n-1)*12+10)
    B_small = B(coord1, coord2);
    imagesc(B_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax4_bis,parula)
    if n==1
        title('(d*)','interpreter','latex','linewidth',9)
    else
        title('(j*)', 'interpreter','latex')
    end
    
    ax5 = subplot(numel(d)*2,6,(n-1)*12+5);
    sol = importdata(fullfile(d2,'Results_depth', 'parameters.mat'));
    sol = sol.result;
    sol = squeeze(sol);
    imagesc(sol)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    caxis([0,1])
    colormap(ax5,parula)
    rmse_val = cal_rmse(sol, df);
    txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n==1
        title(ax5,{'\textbf{DepthSR-Net}','(e)'},'interpreter','latex')
    else
        title('(k)', 'interpreter','latex')
    end

    ax5_bis = subplot(numel(d)*2,6,(n-1)*12+11)
    sol_small = sol(coord1, coord2);
    imagesc(sol_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax5_bis,parula)
    if n==1
        title('(e*)','interpreter','latex','linewidth',9)
    else
        title('(k*)', 'interpreter','latex')
    end

    
    ax6 = subplot(numel(d)*2,6,(n-1)*12+6);
    sol = importdata(fullfile(d2,'Results/Reconstructions', '0recon.mat'));
    imagesc(sol)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    caxis([0,1])
    colormap(ax5,parula)
    rmse_val = cal_rmse(sol, df);
    txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n==1
        title(ax6,{'\textbf{Proposed HistSR-Net}','(f)'},'interpreter','latex')
    else
        title('(l)', 'interpreter','latex')
    end

    ax6_bis = subplot(numel(d)*2,6,(n-1)*12+12)
    sol_small = sol(coord1, coord2);
    imagesc(sol_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax6_bis,parula)
    if n==1
        title('(f*)','interpreter','latex','linewidth',9)
    else
        title('(l*)', 'interpreter','latex')
    end

 
end
set(gcf, 'Position',[118          82        1249         723])
%% Simulated Data without DepthSR-Net
close all
PathName = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Simulate_data/*SBR0.02';
Path = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Simulate_data/'
d = dir(fullfile(PathName));
d = d([d.isdir] & ~ismember({d.name},{'.','..'}));
 
 
for n = 1:numel(d)
    if n == 1
        coord1 = 200:500;
        coord2 = 600:800;
    else 
        coord1 = 200:400;
        coord2 = 200:400;
    end
 
    d2 = fullfile(Path, d(n).name);
    disp(d2)
    
    ax1 = subplot(numel(d)*2,5,(n-1)*10+1);
    df = importdata(fullfile(d2,'data', '0_Df.mat'));
    imagesc(df)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax1,parula)
    if n==1
        title({'\textbf{Ground Truth Depth}','(a)'},'interpreter','latex','linewidth',9)
    else
        title('(k)', 'interpreter','latex')
    end
    
    ax1bis = subplot(numel(d)*2,5,(n-1)*10+6)
    df_small = df(coord1, coord2);
    imagesc(df_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax1bis,parula)
    if n==1
        title('(f)','interpreter','latex','linewidth',9)
    else
        title('(p)', 'interpreter','latex')
    end
    
    ax2 = subplot(numel(d)*2,5,((n-1)*10+2))
    inten = imread(fullfile(d2,'data', '0_RGB.bmp'));
    imagesc(inten)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    colormap(ax2,gray)
    if n == 1
        title({'\textbf{Ground Truth Intensity}','(b)'},'interpreter','latex')
    else
        title('(l)','interpreter','latex')
    end
    
    ax2_bis = subplot(numel(d)*2,5,(n-1)*10+7)
    inten_small = inten(coord1, coord2);
    imagesc(inten_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    colormap(ax2_bis,gray)
    if n==1
        title('(g)','interpreter','latex','linewidth',9)
    else
        title('(q)', 'interpreter','latex')
    end
    
    ax3 = subplot(numel(d)*2,5,(n-1)*10+3);
    interp_image = importdata(fullfile(d2,'data', '0_Df_down.mat'));
    imagesc(interp_image)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    caxis([0,1])
    colormap(ax3,parula)
    rmse_val = cal_rmse(interp_image,df);
    txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n==1
        title({'\textbf{Nearest Interpolation}','(c)'},'interpreter','latex')
    else
        title('(m)', 'interpreter','latex')
    end  
      
    
    
    ax3_bis = subplot(numel(d)*2,5,(n-1)*10+8)
    interp_image_small = interp_image(coord1, coord2);
    imagesc(interp_image_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax3_bis,parula)
    if n==1
        title('(h)','interpreter','latex','linewidth',9)
    else
        title('(r)', 'interpreter','latex')
    end
    
    ax4 = subplot(numel(d)*2,5,(n-1)*10+4)
    B = imguidedfilter(interp_image,inten);
    imagesc(B)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    caxis([0,1])
    colormap(ax4,parula)
    rmse_val = cal_rmse(B,df);
    txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n==1
        title({'\textbf{Guided Filtering}','(d)'},'interpreter','latex')
    else
        title('(n)', 'interpreter','latex')
    end


    ax4_bis = subplot(numel(d)*2,5,(n-1)*10+9)
    B_small = B(coord1, coord2);
    imagesc(B_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax4_bis,parula)
    if n==1
        title('(i)','interpreter','latex','linewidth',9)
    else
        title('(s)', 'interpreter','latex')
    end
    
    ax5 = subplot(numel(d)*2,5,(n-1)*10+5);
    sol = importdata(fullfile(d2,'Results/Reconstructions', '0recon.mat'));
    imagesc(sol)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis image
    axis off
    caxis([0,1])
    colormap(ax5,parula)
    rmse_val = cal_rmse(sol, df);
    txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n==1
        title(ax5,{'\textbf{Proposed HistSR-Net}','(e)'},'interpreter','latex')
    else
        title('(o)', 'interpreter','latex')
    end

    ax5_bis = subplot(numel(d)*2,5,(n-1)*10+10)
    sol_small = sol(coord1, coord2);
    imagesc(sol_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    axis image
    axis off
    caxis([0,1])
    colormap(ax5_bis,parula)
    if n==1
        title('(j)','interpreter','latex','linewidth',9)
    else
        title('(t)', 'interpreter','latex')
    end

 
end
set(gcf, 'Position',[118          82        1249         723])
%% Real data
close all
coord1 = 20:80;
coord2= 100:200;
for n = [1,2,3]
    if n == 1
        Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Hammer_9/frame_9';
    
    else 
        Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Juggling_43'
    end
    
    inten = imread(fullfile(Dir,'0_RGB.bmp'));
    if n ==3
        inten = inten(coord1,coord2);
    end
    ax1 = subplot(3,6,(n-1)*6+1);
    imagesc(inten)
    if n ==2
        hold on
        rectangle('Position',[100,20, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    elseif n ==3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    end
    axis off
    axis image
    colormap(ax1, gray)
    if n ==1
        title({'\textbf{Intensity image}','(a)'},'interpreter','latex')
    elseif n == 2
        title('(g)','interpreter','latex')
    elseif n == 3
        title('(g*)','interpreter','latex')
    end    
    interp_image = importdata(fullfile(Dir,'0_Df_down.mat'));
    if n ==3
        interp_image = interp_image(coord1,coord2);
    end
    
    ax2 = subplot(3,6,(n-1)*6+2)
    imagesc(interp_image)
    if n ==2
        hold on
        rectangle('Position',[100,20, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    elseif n ==3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    end
    caxis([0,1])
    axis off
    axis image
    colormap(ax2, parula)
    if n == 1
        title({'\textbf{Nearest Interpolation}', '(b)'},'interpreter','latex')
    elseif n ==2 
        title('(h)','interpreter','latex')
    elseif n == 3
        title('(h*)','interpreter','latex')
    end
    ax3 = subplot(3,6,(n-1)*6+3)
    B = imguidedfilter(interp_image,inten);
    
    imagesc(B)
    if n ==2
        hold on
        rectangle('Position',[100,20, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
        
    elseif n ==3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    end
    axis off
    caxis([0,1])
    axis image
    colormap(ax3, parula)
    if n ==1
        title({'\textbf{Guided Image Filtering}','(c)'},'interpreter','latex')
    elseif n == 2
        title('(i)','interpreter','latex')
    elseif n == 3
        title('(i*)','interpreter','latex')
    end
    
    ax4 =subplot(3,6,(n-1)*6+4)
    if n == 1
        res = importdata('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Hammer_9/frame_9/DepthNet/result.mat')% importdata(fullfile(Dir,'DepthNet','result.mat'));
    else
        res = importdata('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Juggling_43/DepthNet/result.mat')% importdata(fullfile(Dir,'DepthNet','result.mat'));
    end
        
    if n == 3
        res = res(coord1,coord2);
    end
    imagesc(res)
    if n ==2
        hold on
        rectangle('Position',[100,20, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    elseif n ==3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    end
    axis off
    caxis([0,1])
    colormap(ax4, parula)
    axis image
    if n == 1
        title({'\textbf{Guo \textit{et al.} 2019}','(d)'},'interpreter','latex')
    elseif n == 2
        title('(j)','interpreter','latex')
    elseif n == 3
        title('(j*)','interpreter','latex')
    end
    
    ax5 =subplot(3,6,(n-1)*6+5)
    if n == 1
        res = importdata('/Users/aliceruget/Documents/PhD/Optica/data_LAB_JONATHAN_ALICE_TEST_FUSION_REAL.mat');% importdata(fullfile(Dir,'DepthNet','result.mat'));
        res = res.D_processed(:,:,3);
    else
        res = importdata('/Users/aliceruget/Documents/PhD/Optica/data_LAB_JONATHAN_ALICE_TEST_FUSION_REAL.mat');% importdata(fullfile(Dir,'DepthNet','result.mat'));
        res = res.D_processed(:,:,1);
    end
        
    if n == 3
        res = res(coord1,coord2);
    end
    imagesc(res)
    if n ==2
        hold on
        rectangle('Position',[100,20, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    elseif n ==3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    end
    axis off
    caxis([0,1])
    colormap(ax4, parula)
    axis image
    if n == 1
        title({'\textbf{Gyongy \textit{et al.} 2020}','(e)'},'interpreter','latex')
    elseif n == 2
        title('(k)','interpreter','latex')
    elseif n == 3
        title('(k*)','interpreter','latex')
    end
    
    ax6 =subplot(3,6,(n-1)*6+6)
    %Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/real_data/Hammer/DATA_TEST_otherlevel_17_frame_9';
    if n == 1
        res = importdata('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/real_data/Hammer/DATA_TEST_otherlevel_17_frame_9/res.mat');
    else
        res = importdata(fullfile(Dir,'TwoDepthNet','Reconstructions','0recon.mat'));
    end
    
    if n == 3
        res = res(coord1,coord2);
    end
    
    imagesc(res)
    if n ==2
        hold on
        rectangle('Position',[100,20, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    elseif n ==3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    end
    axis off
    caxis([0,1])
    colormap(ax5, parula)
    axis image
    
    
    
    
    if n == 1
        cb = colorbar
        cb.Ticks = [0,0.5,1];
        cb.TickLabels = {0,1,2};
        set(get(cb,'label'),'string','Depth in m')
        cb.Position = [0.91832229580574,0.113095238095238,0.003311258278146,0.785714285714286];
    end
    if n == 1
        title({'\textbf{Proposed HistNet}','(f)'},'interpreter','latex')
    elseif n == 2
        title('(l)','interpreter','latex')
    elseif n == 3
        title('(l*)','interpreter','latex')
    end
end
set(gcf,'Position',[144   335   985   336])

%% Real data without OPtica
close all
coord1 = 20:80;
coord2= 100:200;
for n = [1,2,3]
    if n == 1
        Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Hammer_9/frame_9';
    
    else 
        Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Juggling_43'
    end
    
    inten = imread(fullfile(Dir,'0_RGB.bmp'));
    if n ==3
        inten = inten(coord1,coord2);
    end
    ax1 = subplot(3,5,(n-1)*5+1);
    imagesc(inten)
    if n ==2
        hold on
        rectangle('Position',[100,20, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    elseif n ==3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    end
    axis off
    axis image
    colormap(ax1, gray)
    if n ==1
        title({'\textbf{Intensity image}','(a)'},'interpreter','latex')
    elseif n == 2
        title('(f)','interpreter','latex')
    elseif n == 3
        title('(k)','interpreter','latex')
    end    
    interp_image = importdata(fullfile(Dir,'0_Df_down.mat'));
    if n ==3
        interp_image = interp_image(coord1,coord2);
    end
    
    ax2 = subplot(3,5,(n-1)*5+2)
    imagesc(interp_image)
    if n ==2
        hold on
        rectangle('Position',[100,20, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    elseif n ==3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    end
    caxis([0,1])
    axis off
    axis image
    colormap(ax2, parula)
    if n == 1
        title({'\textbf{Nearest Interpolation}', '(b)'},'interpreter','latex')
    elseif n ==2 
        title('(g)','interpreter','latex')
    elseif n == 3
        title('(l)','interpreter','latex')
    end
    ax3 = subplot(3,5,(n-1)*5+3)
    B = imguidedfilter(interp_image,inten);
    
    imagesc(B)
    if n ==2
        hold on
        rectangle('Position',[100,20, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
        
    elseif n ==3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    end
    axis off
    caxis([0,1])
    axis image
    colormap(ax3, parula)
    if n ==1
        title({'\textbf{Guided Filtering}','(c)'},'interpreter','latex')
    elseif n == 2
        title('(h)','interpreter','latex')
    elseif n == 3
        title('(m)','interpreter','latex')
    end
    
    ax4 =subplot(3,5,(n-1)*5+4)
    if n == 1
        res = importdata('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Hammer_9/frame_9/DepthNet/result.mat')% importdata(fullfile(Dir,'DepthNet','result.mat'));
    else
        res = importdata('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Juggling_43/DepthNet/result.mat')% importdata(fullfile(Dir,'DepthNet','result.mat'));
    end
        
    if n == 3
        res = res(coord1,coord2);
    end
    
    imagesc(res)
    if n ==2
        hold on
        rectangle('Position',[100,20, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    elseif n ==3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    end
    axis off
    caxis([0,1])
    colormap(ax4, parula)
    axis image
    if n == 1
        title({'\textbf{DepthSR-Net}','(d)'},'interpreter','latex')
    elseif n == 2
        title('(i)','interpreter','latex')
    elseif n == 3
        title('(n)','interpreter','latex')
    end
    
    
    ax5 =subplot(3,5,(n-1)*5+5)
    %Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/real_data/Hammer/DATA_TEST_otherlevel_17_frame_9';
    if n == 1
        res = importdata('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/real_data/Hammer/DATA_TEST_otherlevel_17_frame_9/res.mat');
    else
        res = importdata(fullfile(Dir,'TwoDepthNet','Reconstructions','0recon.mat'));
    end
    
    if n == 3
        res = res(coord1,coord2);
    end
    
    imagesc(res)
    if n ==2
        hold on
        rectangle('Position',[100,20, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    elseif n ==3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
    end
    axis off
    caxis([0,1])
    colormap(ax5, parula)
    axis image
    
    
    
    
    if n == 1
        cb = colorbar
        cb.Ticks = [0,0.5,1];
        cb.TickLabels = {0,1,2};
        set(get(cb,'label'),'string','Depth in m')
        cb.Position = [0.91832229580574,0.113095238095238,0.003311258278146,0.785714285714286];
    end
    if n == 1
        title({'\textbf{Proposed HistSR-Net}','(e)'},'interpreter','latex')
    elseif n == 2
        title('(j)','interpreter','latex')
    elseif n == 3
        title('(o)','interpreter','latex')
    end
end
set(gcf,'Position',[144   335   906   336])


%% Comparaison w/o Second Depth Simulated and Real Data
close all
for n = [1,2]
    if n == 1
            coord1 = 200:500;
            coord2 = 600:800;
            Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
        else 
            coord1 = 800:1100;
            coord2 = 200:400;
            Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_geo';     
    end

    %Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
    df = importdata(fullfile(Dir,'0_Df.mat'));

    ax1 = subplot(4,5,(n-1)*10+1)
    imagesc(df)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    %axis off
    caxis([0,1])
    colormap(ax1, parula)
    axis image
    if n ==1
        title({'Ground Truth','(a)'},'interpreter', 'latex')
    else
        title('(k)','interpreter', 'latex')
    end
    
    ax1_bis = subplot(4,5,(n-1)*10+6)
    df_small = df(coord1,coord2);
    imagesc(df_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r')
    axis off
    caxis([0,1])
    colormap(ax1, parula)
    axis image
    if n == 1
        title('(f)','interpreter', 'latex')
    else
        title('(p)','interpreter', 'latex')
    end
    ax2 = subplot(4,5,(n-1)*10+2)
    df_down = importdata(fullfile(Dir,'0_Df_down.mat'));
    imagesc(df_down)
    axis off
    axis image
    caxis([0,1])
    colormap(ax2, parula)
    if n == 1
        title({'First Depth','(b)'},'interpreter', 'latex')
    else
        title('(l)','interpreter', 'latex')
    end
    
    rmse_val = cal_rmse(df_down, df);
    txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)

    ax2_bis = subplot(4,5,(n-1)*10+7)
    df_down_small = df_down(coord1,coord2);
    imagesc(df_down_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r')
    axis off
    axis image
    caxis([0,1])
    colormap(ax2_bis, parula)
    if n ==1 
        title('(g)','interpreter', 'latex')
    else
        title('(q)','interpreter', 'latex')
    end


    ax3 = subplot(4,5,(n-1)*10+3)
    df_down_2 =importdata(fullfile(Dir,'0_Df_down_2.mat'));
    imagesc(df_down_2)
    axis off
    axis image
    caxis([0,1])
    colormap(ax3, parula)
    if n == 1
        title({'Second Depth','(c)'},'interpreter', 'latex')
    else
        title('(m)','interpreter', 'latex')
    end

    ax3_bis = subplot(4,5,(n-1)*10+8)
    df_down_2 = df_down_2(coord1,coord2);
    imagesc(df_down_2)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r')
    axis off
    axis image
    caxis([0,1])
    colormap(ax3_bis, parula)
    if n == 1
        title('(h)','interpreter', 'latex')
    else
        title('(r)','interpreter', 'latex')
    end
        

    ax4 = subplot(4,5,(n-1)*10+4)
    resw0 = squeeze(importdata(fullfile(Dir,'HistNet', 'result.mat')));
    imagesc(squeeze(resw0))
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis off
    axis image
    caxis([0,1])
    colormap(ax4, parula)
    rmse_val = cal_rmse(resw0, df);
    txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n ==1
        title({'Proposed HistSR-Net', 'w/0 Second Depth','(d)'},'interpreter', 'latex')
    else
        title('(n)','interpreter', 'latex')
    end

    ax4_bis = subplot(4,5,(n-1)*10+9)
    resw0_small = resw0(coord1,coord2);
    imagesc(resw0_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r')
    axis off
    axis image
    caxis([0,1])
    colormap(ax2_bis, parula)
    if n == 1
        title('(i)','interpreter', 'latex')
    else
        title('(s)','interpreter', 'latex')
    end
    ax5 = subplot(4,5,(n-1)*10+5)
    %resw = squeeze(importdata('/Users/aliceruget/Documents/PhD/HistSR_Net_AR/Results/i3000_SBR_0.4/Middlebury_test/Art/HistNet/result.mat'));
    resw =importdata(fullfile(Dir,'TwoDepthNet','result.mat'));
    imagesc(resw)
    hold on
    rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis off
    axis image
    caxis([0,1])
    colormap(ax4, parula)
    rmse_val = cal_rmse(resw, df);
    txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n == 1
        title({'Proposed HistSR-Net', 'w Second Depth','(e)'},'interpreter', 'latex')
    else
        title('(o)', 'interpreter', 'latex')
    end
    ax5_bis = subplot(4,5,(n-1)*10+10)
    resw_small = resw(coord1,coord2);
    imagesc(resw0_small)
    hold on
    rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r')
    axis off
    axis image
    caxis([0,1])
    colormap(ax2_bis, parula)
    if n == 1
        title('(j)','interpreter', 'latex')
    else
        title('(t)','interpreter', 'latex')
    end
    
end


set(gcf,'Position', [ 109   378   891   420])


%% Comparaison w/o Second Depth Simulated and Real Data
% NO SMALL FIGURE
close all
for n = [1,2]
    if n == 1
            coord1 = 200:500;
            coord2 = 600:800;
            Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
        else 
            coord1 = 800:1100;
            coord2 = 200:400;
            Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_geo';     
    end

    %Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
    df = importdata(fullfile(Dir,'0_Df.mat'));

    ax1 = subplot(2,5,(n-1)*5+1)
    imagesc(df)
    %hold on
    %rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis off
    caxis([0,1])
    colormap(ax1, parula)
    axis image
    if n ==1
        title({'\textbf{Ground Truth}','(a)'},'interpreter', 'latex')
    else
        title('(f)','interpreter', 'latex')
    end

    ax2 = subplot(2,5,(n-1)*5+2)
    df_down = importdata(fullfile(Dir,'0_Df_down.mat'));
    imagesc(df_down)
    axis off
    axis image
    caxis([0,1])
    colormap(ax2, parula)
    if n == 1
        title({'\textbf{First Depth}','(b)'},'interpreter', 'latex')
    else
        title('(g)','interpreter', 'latex')
    end
    
    rmse_val = cal_rmse(df_down, df);
    txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)

    

    ax3 = subplot(2,5,(n-1)*5+3)
    df_down_2 =importdata(fullfile(Dir,'0_Df_down_2.mat'));
    imagesc(df_down_2)
    axis off
    axis image
    caxis([0,1])
    colormap(ax3, parula)
    if n == 1
        title({'\textbf{Second Depth}','(c)'},'interpreter', 'latex')
    else
        title('(h)','interpreter', 'latex')
    end

    
        

    ax4 = subplot(2,5,(n-1)*5+4)
    resw0 = squeeze(importdata(fullfile(Dir,'HistNet', 'result.mat')));
    imagesc(squeeze(resw0))
    %hold on
    %rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis off
    axis image
    caxis([0,1])
    colormap(ax4, parula)
    rmse_val = cal_rmse(resw0, df);
    txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n ==1
        title({'\textbf{Proposed HistSR-Net}', '\textbf{w/0 Second Depth}','(d)'},'interpreter', 'latex')
    else
        title('(i)','interpreter', 'latex')
    end

    
    ax5 = subplot(2,5,(n-1)*5+5)
    %resw = squeeze(importdata('/Users/aliceruget/Documents/PhD/HistSR_Net_AR/Results/i3000_SBR_0.4/Middlebury_test/Art/HistNet/result.mat'));
    resw =importdata(fullfile(Dir,'TwoDepthNet','result.mat'));
    imagesc(resw)
    %hold on
    %rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    axis off
    axis image
    caxis([0,1])
    colormap(ax4, parula)
    rmse_val = cal_rmse(resw, df);
    txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
    text(10,70,txt,'Color','white', 'Fontsize',12)
    if n == 1
        title({'\textbf{Proposed HistSR-Net}', '\textbf{w Second Depth}','(e)'},'interpreter', 'latex')
    else
        title('(j)', 'interpreter', 'latex')
    end
    
    
end


set(gcf,'Position', [ 109         244        1104         398])


%% COMPA DepthSR-Net Not NOISY
close all

%2DepthNet
N = 3;
%for n = 1:N
for n =4:6
    if n ==4
        n_index = 1
    elseif n ==5 
        n_index = 2
    elseif n ==6 
        n_index = 3
    end
    if n == 1
        Dir_saved = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
        Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
        Dir_down = '/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Middlebury_art_i3000_sbr_4/0_Df_down.mat'
        %Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i10_SBR_0_0004_l1/Middlebury_art';
%     elseif n == 2
%         Dir_saved = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_geo';
%         Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_geo';
%         %Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i10_SBR_0_0004_l1/Middlebury_geo';
    elseif n == 2
        Dir_saved = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
        Dir_down = '/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Middlebury_art_i10_sbr_004/0_Df_down.mat';
        Dir ='/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i10_SBR_0_0004_l1/Middlebury_art';
        coord1 = 200:500;
        coord2 = 600:800;
    elseif n == 3
        Dir_saved = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
        Dir_down = '/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Middlebury_art_i10_sbr_004/0_Df_down.mat';
        Dir ='/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i10_SBR_0_0004_l1/Middlebury_art';
        coord1 = 200:500;
        coord2 = 600:800;
%     elseif n == 4 
%         Dir_saved = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_geo';
%         Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i10_SBR_0_0004_l1/Middlebury_geo';
     elseif n == 4
        Dir_down = '/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Hammer_real/frame_9/0_Df_down.mat'
        Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Hammer_9/frame_9';
    elseif n == 5
        Dir_down = '/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Juggling_real/frame_40/40_Df_down.mat'
        Dir ='/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Juggling_43';
        coord1 = 20:80;
        coord2= 100:200;
    elseif n ==6
        Dir_down = '/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Juggling_real/frame_40/40_Df_down.mat'
        Dir ='/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Juggling_43';
        coord1 = 20:80;
        coord2= 100:200;
    
    end
       
    %Dir_bis = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
    if n==1|n==2|n == 3
        Df = importdata(fullfile(Dir_saved, '0_Df.mat'));
    end
        
    Df_down= importdata(Dir_down);
    
    if n ==6
        Df_down = Df_down(coord1,coord2);
    elseif n == 3
        Df_down = Df_down(coord1,coord2);
    end
    
    subplot(N,3,(n_index-1)*3+1)
    imagesc(Df_down)
    
    if n ==5 | n == 2
        hold on
        rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    elseif n ==6 | n == 3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',2)  
    end
    
    axis image
    if n ==1 | n ==2
        caxis([0,1])
    end
    if n == 1
        title({'\textbf{First Depth Input}','(a)'}, 'interpreter', 'latex')
        ylabel({'\textbf{Simulated Data}', 'SBR=2 and ppp=1200'},'interpreter', 'latex')
    elseif n==2
        title('(d)', 'interpreter', 'latex')
        ylabel({'\textbf{Simulated Data}', 'SBR=0.02 and ppp=4'},'interpreter', 'latex')
    elseif n==3
        title('(g)', 'interpreter', 'latex')
        ylabel({'\textbf{Closeup Simulated Data}', 'SBR=0.02 and ppp=4'},'interpreter', 'latex')
       
    elseif n==4
        title({'\textbf{First Depth Input}','(a)'}, 'interpreter', 'latex')
        ylabel('\textbf{Captured Data}','interpreter', 'latex')
        %ylabel({'\textbf{Captured Data}', 'SBR=2 and ppp=1200'},'interpreter', 'latex')
    elseif n ==5
        title('(d)', 'interpreter', 'latex')
        ylabel('\textbf{Captured Data}','interpreter', 'latex')
    elseif n ==6
        title('(g)', 'interpreter', 'latex')
        ylabel('\textbf{Closeup Captured Data}','interpreter', 'latex')
            
    end
    if n == 1 | n ==2
        rmse_val = cal_rmse(Df_down, Df);
        txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
        text(10,70,txt,'Color','white', 'Fontsize',12)
    end
    
    subplot(N,3,(n_index-1)*3+2)
    res = importdata(fullfile(Dir, 'DepthNet', 'result.mat'));
    if n ==6
        res = res(coord1, coord2);
    elseif n == 3
        res = res(coord1, coord2);
    end 
    imagesc(res)
    if n ==5 | n == 2
        hold on
        rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    elseif n ==6 | n == 3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',2)  
    end
    %axis off
    if n ==1 | n ==2
        caxis([0,1])
    end
    axis image
    if n == 1 | n ==2
        rmse_val = cal_rmse(res, Df);
        txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
        text(10,70,txt,'Color','white', 'Fontsize',12)
    end
    if n == 1
        title({'\textbf{DepthSR-Net}','(b)'}, 'interpreter', 'latex')
    elseif n==2
        title('(e)', 'interpreter', 'latex')
    elseif n==3
        title('(h)', 'interpreter', 'latex')
    elseif n==4
        title({'\textbf{DepthSR-Net}','(b)'}, 'interpreter', 'latex')
    elseif n==5
        title('(e)', 'interpreter', 'latex')
    elseif n==6
        title('(h)', 'interpreter', 'latex')
    end
    subplot(N,3,(n_index-1)*3+3)  
    res = importdata(fullfile(Dir, 'TwoDepthNet', 'result.mat'));
    if n == 6
        res =res(coord1, coord2);
    elseif n == 3
        res =res(coord1, coord2);
    end
    imagesc(res)
    if n == 2 
        hold on
        rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    elseif n == 5
        hold on
        rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    elseif n == 3 
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',2)  
    elseif n == 6
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',2)  
    end
    %axis off
    axis image
    %caxis([0,1])
    if n == 1 | n ==2
        rmse_val = cal_rmse(res, Df);
        txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
        text(10,70,txt,'Color','white', 'Fontsize',12)
    end
    if n == 1
        title({'\textbf{HistSR-Net}','(c)'}, 'interpreter', 'latex')
    elseif n==2
        title('(f)', 'interpreter', 'latex')
    elseif n==3
        title('(i)', 'interpreter', 'latex')
    elseif n==4
        title({'\textbf{HistSR-Net}','(c)'}, 'interpreter', 'latex')
    elseif n==5
        title('(f)', 'interpreter', 'latex')    
    elseif n==6
        title('(i)', 'interpreter', 'latex') 
    end

end
%set(gcf,'Position', [44         352        1284         447])
set(gcf,'Position', [319    95   881   710])

%% COMPA DEPTHSR-NET NOISY
close all
%2DepthNet
N = 2;
for n = 1:N
    if n == 1
        Dir_bis = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';

        Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i10_SBR_0_0004_l1/Middlebury_art';
    elseif n ==2
        Dir_bis = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_geo';
        Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i10_SBR_0_0004_l1/Middlebury_geo';
    end
       
    %Dir_bis = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
    Df = importdata(fullfile(Dir_bis, '0_Df.mat'));
    
    subplot(2,N,n)
    res = importdata(fullfile(Dir, 'DepthNet', 'result.mat'));
    imagesc(res)
    axis off
    %caxis([0,1])
    axis image
    if n == 1 | n ==2
        rmse_val = cal_rmse(res, Df);
        txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
        text(10,70,txt,'Color','white', 'Fontsize',12)
    end
    if n==1 
        title('\textbf{(a) DepthSR-Net applied to simulated data with SNR = 0.02 and ppp = 4}','interpreter','latex', 'linewidth',15)
    end

    subplot(2,N,n+N)
    res = importdata(fullfile(Dir, 'TwoDepthNet', 'result.mat'));
    imagesc(res)
    axis off
    axis image
    %caxis([0,1])
    if n == 1 | n ==2
        rmse_val = cal_rmse(res, Df);
        txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
        text(10,70,txt,'Color','white', 'Fontsize',12)
    end
    if n==1 
        title('\textbf{(b) HistSR-Net applied to simulated data with SNR = 0.02 and ppp = 4}','interpreter','latex','linewidth',15)
    end

end
%set(gcf,'Position', [44         352        1284         447])
set(gcf,'Position', [221   380   461   357])
%% COMPA DepthSR-Net NOISY
close all

%2DepthNet
N = 3;
for n = 1:N
%for n =4:6
    if n ==1
        n_index = 1
    elseif n ==2
        n_index = 2
    elseif n ==3
        n_index = 3
    end
    if n == 1
        Dir_saved = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
        Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
        Dir_down = '/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Middlebury_art_i3000_sbr_4/0_Df_down.mat'
        %Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i10_SBR_0_0004_l1/Middlebury_art';
%     elseif n == 2
%         Dir_saved = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_geo';
%         Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_geo';
%         %Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i10_SBR_0_0004_l1/Middlebury_geo';
    elseif n == 2
        Dir_saved = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
        Dir_down = '/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Middlebury_art_i10_sbr_004/0_Df_down.mat';
        Dir ='/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i10_SBR_0_0004_l1/Middlebury_art';
        coord1 = 200:500;
        coord2 = 600:800;
    elseif n == 3
        Dir_saved = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
        Dir_down = '/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Middlebury_art_i10_sbr_004/0_Df_down.mat';
        Dir ='/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i10_SBR_0_0004_l1/Middlebury_art';
        coord1 = 200:500;
        coord2 = 600:800;
%     elseif n == 4 
%         Dir_saved = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_geo';
%         Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i10_SBR_0_0004_l1/Middlebury_geo';
     elseif n == 4
        Dir_down = '/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Hammer_real/frame_9/0_Df_down.mat'
        Dir = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Hammer_9/frame_9';
    elseif n == 5
        Dir_down = '/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Juggling_real/frame_40/40_Df_down.mat'
        Dir ='/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Juggling_43';
        coord1 = 20:80;
        coord2= 100:200;
    elseif n ==6
        Dir_down = '/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Juggling_real/frame_40/40_Df_down.mat'
        Dir ='/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Juggling_43';
        coord1 = 20:80;
        coord2= 100:200;
    
    end
       
    %Dir_bis = '/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/Results/i3000_SBR_04/Middlebury_art';
    if n==1|n==2|n == 3
        Df = importdata(fullfile(Dir_saved, '0_Df.mat'));
    end
        
    Df_down= importdata(Dir_down);
    
    if n ==6
        Df_down = Df_down(coord1,coord2);
    elseif n == 3
        Df_down = Df_down(coord1,coord2);
    end
    
    subplot(N,3,(n_index-1)*3 + 1)
    imagesc(Df_down)
    
    if n ==5 | n == 2
        hold on
        rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    elseif n ==6 | n == 3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',2)  
    end
    
    axis image
    if n ==1 | n ==2
        caxis([0,1])
    end
    if n == 1
        title({'\textbf{First Depth Input}','(a)'}, 'interpreter', 'latex')
        ylabel({'\textbf{Simulated Data}', 'SBR=2 and ppp=1200'},'interpreter', 'latex')
    elseif n==2
        title('(d)', 'interpreter', 'latex')
        ylabel({'\textbf{Simulated Data}', 'SBR=0.02 and ppp=4'},'interpreter', 'latex')
    elseif n==3
        title('(g)', 'interpreter', 'latex')
        ylabel({'\textbf{Closeup Simulated Data}', 'SBR=0.02 and ppp=4'},'interpreter', 'latex')
       
    elseif n==4
        title({'\textbf{First Depth Input}','(a)'}, 'interpreter', 'latex')
        ylabel('\textbf{Captured Data}','interpreter', 'latex')
        %ylabel({'\textbf{Captured Data}', 'SBR=2 and ppp=1200'},'interpreter', 'latex')
    elseif n ==5
        title('(d)', 'interpreter', 'latex')
        ylabel('\textbf{Captured Data}','interpreter', 'latex')
    elseif n ==6
        title('(g)', 'interpreter', 'latex')
        ylabel('\textbf{Closeup Captured Data}','interpreter', 'latex')
            
    end
    if n == 1 | n ==2
        rmse_val = cal_rmse(Df_down, Df);
        txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
        text(10,70,txt,'Color','white', 'Fontsize',12)
    end
    
    subplot(N,3,(n_index-1)*3+2)
    res = importdata(fullfile(Dir, 'DepthNet', 'result.mat'));
    if n ==6
        res = res(coord1, coord2);
    elseif n == 3
        res = res(coord1, coord2);
    end 
    imagesc(res)
    caxis([0,1])
    if n ==5 | n == 2
        hold on
        rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    elseif n ==6 | n == 3
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',2)  
    end
    %axis off
    if n ==1 | n ==2
        caxis([0,1])
    end
    axis image
    if n == 1 | n ==2
        rmse_val = cal_rmse(res, Df);
        txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
        text(10,70,txt,'Color','white', 'Fontsize',12)
    end
    if n == 1
        title({'\textbf{DepthSR-Net}','(b)'}, 'interpreter', 'latex')
    elseif n==2
        title('(e)', 'interpreter', 'latex')
    elseif n==3
        title('(h)', 'interpreter', 'latex')
    elseif n==4
        title({'\textbf{DepthSR-Net}','(b)'}, 'interpreter', 'latex')
    elseif n==5
        title('(e)', 'interpreter', 'latex')
    elseif n==6
        title('(h)', 'interpreter', 'latex')
    end
    subplot(N,3,(n_index-1)*3+3)  
    res = importdata(fullfile(Dir, 'TwoDepthNet', 'result.mat'));
    if n == 6
        res =res(coord1, coord2);
    elseif n == 3
        res =res(coord1, coord2);
    end
    imagesc(res)
    caxis([0,1])
    if n == 2 
        hold on
        rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    elseif n == 5
        hold on
        rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r')
    elseif n == 3 
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',2)  
    elseif n == 6
        hold on
        rectangle('Position',[1,1, length(coord2), length(coord1)],'EdgeColor','r','linewidth',2)  
    end
    %axis off
    axis image
    %caxis([0,1])
    if n == 1 | n ==2
        rmse_val = cal_rmse(res, Df);
        txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
        text(10,70,txt,'Color','white', 'Fontsize',12)
    end
    if n == 1
        title({'\textbf{HistSR-Net}','(c)'}, 'interpreter', 'latex')
    elseif n==2
        title('(f)', 'interpreter', 'latex')
    elseif n==3
        title('(i)', 'interpreter', 'latex')
    elseif n==4
        title({'\textbf{HistSR-Net}','(c)'}, 'interpreter', 'latex')
    elseif n==5
        title('(f)', 'interpreter', 'latex')    
    elseif n==6
        title('(i)', 'interpreter', 'latex') 
    end

end
%set(gcf,'Position', [44         352        1284         447])
set(gcf,'Position', [319    95   881   710])

%% HISTOGRAM CREATION DATASET
close all
for i = 1:8
    if i == 1
        A1 = imread('/Users/aliceruget/Documents/PhD/Dataset/Methods_and_Codes/HIstogram-figures/histo_fig.png');
        A1 = imcrop(A1,[127.5 67.5 401 331]); 
    elseif i == 2
        A1 = imread('/Users/aliceruget/Documents/PhD/Dataset/Methods_and_Codes/HIstogram-figures/DS2_hist.png');
        A1 = imcrop(A1,[127.5 67.5 401 331]); 
    elseif i == 3
        A1 = imread('/Users/aliceruget/Documents/PhD/Dataset/Methods_and_Codes/HIstogram-figures/DS3_hist.png');
        A1 = imcrop(A1,[127.5 67.5 401 331]); 
    elseif i == 4
        A1 = imread('/Users/aliceruget/Documents/PhD/Dataset/Methods_and_Codes/HIstogram-figures/DS4_hist.png');
        A1 = imcrop(A1,[127.5 67.5 401 331]); 
    elseif i == 5
        A1 = importdata('/Users/aliceruget/Documents/PhD/Dataset/Hammer_data/Data_Test/9_pool1.mat');
   elseif i == 6
        A1 = importdata('/Users/aliceruget/Documents/PhD/Dataset/Hammer_data/Data_Test/9_pool2.mat');
   elseif i == 7
        A1 = importdata('/Users/aliceruget/Documents/PhD/Dataset/Hammer_data/Data_Test/9_pool3.mat');
   elseif i == 8
        A1 = importdata('/Users/aliceruget/Documents/PhD/Dataset/Hammer_data/Data_Test/9_pool4.mat');
   elseif i == 9
        A1 = imread('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Middlebury_0_ppp4_SBR0.02/Figures/DS2_histo.png');
        A1 = imcrop(A1,[127.5 67.5 401 331]);
   elseif i == 10
        A1 = imread('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Middlebury_0_ppp4_SBR0.02/Figures/DS2_histo.png');
        A1 = imcrop(A1,[127.5 67.5 401 331]);  
   elseif i == 11
        A1 = imread('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Middlebury_0_ppp4_SBR0.02/Figures/DS3_histo.png');
        A1 = imcrop(A1,[127.5 67.5 401 331]);   
  elseif i == 12
        A1 = imread('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Middlebury_0_ppp4_SBR0.02/Figures/DS4_histo.png');
        A1 = imcrop(A1,[127.5 67.5 401 331]);
  elseif i == 13
        A1 = importdata('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Middlebury_0_ppp4_SBR0.02/Figures/DS2_image.mat');
  elseif i == 14
        A1 = importdata('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Middlebury_0_ppp4_SBR0.02/Figures/DS2_image.mat');
  elseif i == 15
        A1 = importdata('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Middlebury_0_ppp4_SBR0.02/Figures/DS3_image.mat');
  elseif i == 16
        A1 = importdata('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/Middlebury_0_ppp4_SBR0.02/Figures/DS4_image.mat');
        A1=squeeze(A1);
    end
    
    subplot(2,4,i)
    imagesc(A1)
    
    axis image
    
   if i == 1
        title('(a)', 'interpreter', 'latex')
        %ylabel('\textbf{Captured Data : Histogram}','interpreter', 'latex')
        h=ylabel({'\textbf{Captured}'}','interpreter', 'latex')
        axesoffwithlabels(h)
    elseif i == 2
        title('(b)', 'interpreter', 'latex')
        axis off
    elseif i == 3
        title('(c)', 'interpreter', 'latex')
        axis off
    elseif i == 4
        title('(d)', 'interpreter', 'latex')
        axis off
    elseif i == 5
        title('(e)', 'interpreter', 'latex')
        axis image
        caxis([0,1])
        axis off
        %h(1)=ylabel({'\textbf{Captured}', 'Depth Features'}','interpreter', 'latex')
   elseif i == 6
       title('(f)', 'interpreter', 'latex')
       axis image
       caxis([0,1])
       %colorbar
       axis off
   elseif i == 7
       title('(g)', 'interpreter', 'latex')
       axis image
       caxis([0,1])
       %colorbar
       axis off
   elseif i == 8
       title('(h)', 'interpreter', 'latex')
       axis image   
       caxis([0,1])
       axis off
   elseif i == 9
        title('(i)', 'interpreter', 'latex')
        axis image
        h = ylabel({'\textbf{Simulated}'}','interpreter', 'latex', 'Fontsize', 12)
        axesoffwithlabels(h)
    elseif i == 10
        title('(j)', 'interpreter', 'latex')
        axis image
        axis off
    elseif i == 11
        title('(k)', 'interpreter', 'latex')
        axis image
        axis off
    elseif i == 12
        title('(l)', 'interpreter', 'latex')
        axis image
        axis off
    elseif i == 13
        title('(m)', 'interpreter', 'latex')
        axis image
        axis off
        %ylabel({'\textbf{Simulated}', 'Depth Features'}','interpreter', 'latex')
        caxis([0,15])
   elseif i == 14
       title('(n)', 'interpreter', 'latex')
       axis image
       caxis([0,15])
       axis off
        
   elseif i == 15
       title('(o)', 'interpreter', 'latex')
       axis image
       caxis([0,15])
       axis off
   elseif i == 16
       title('(p)', 'interpreter', 'latex')
       axis image
       axis off
       caxis([0,15])
   end
    if i == 1
        cb = colorbar
        cb.Ticks = [0,0.5,1];
        cb.TickLabels = {0,1,2};
        set(get(cb,'label'),'string','Depth in m')
        cb.Position = [0.917607223476298,0.556318681318681,0.003386004514672,0.124532382511106];
    end
    if i == 3
        cb = colorbar
        cb.Ticks = [0,0.5,1];
        cb.TickLabels = {0,0.5, 1};
        set(get(cb,'label'),'string','Depth')
        cb.Position = [0.873589164785553,0.111896348645465,0.003386004514673,0.145195849936095];
    end
end
%ha=get(gcf,'children');
%set(ha(1),'position',[0.1 .1 .4 .4])
%set(ha(2),'position',[.1 .1 .4 .4])
%set(ha(3),'position',[.5 .5 .4 .4])
%set(ha(4),'position',[.1 .5 .4 .4])

set(gcf, 'Position', [203   148   886   564])
axesoffwithlabels(h)
%% Second depth figure Explanation 
close all
A1 = imread('/Users/aliceruget/Documents/PhD/Dataset/Methods_and_Codes/HIstogram-figures/first_depth.png');
A1 = imcrop(A1,[127.5 67.5 401 331]); 
A2 = imread('/Users/aliceruget/Documents/PhD/Dataset/Methods_and_Codes/HIstogram-figures/second_depth.png');
A2 = imcrop(A2,[127.5 67.5 401 331]); 
A3 = importdata('/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Hammer_real/frame_9/0_Df_down.mat');
A4 = importdata('/Users/aliceruget/Documents/PhD/Dataset/DATA_TEST/Hammer_real/frame_9/0_Df_down_2.mat');

subplot(2,2,1)
imagesc(A1)
title('(a)', 'interpreter', 'latex')
axis off
subplot(2,2,2)
imagesc(A2)
title('(b)', 'interpreter', 'latex')
axis off
subplot(2,2,3)
imagesc(A3)
title('(c)', 'interpreter', 'latex')
axis off
axis image
caxis([0,1])
subplot(2,2,4)
imagesc(A4)
title('(d)', 'interpreter', 'latex')
axis off
caxis([0,1])
axis image
cb = colorbar
cb.Ticks = [0,0.5,1];
cb.TickLabels = {0,1,2};
set(get(cb,'label'),'string','Depth in m')
cb.Position = [0.921460788023504,0.147692307692308,0.009175050126785,0.265466448445172];

set(gcf, 'Position', [438   405   519   325])

%% Compa Lindell
close all
coord1 = 100:350;
coord2 = 300:450;
axis_colorbar = [1.4357    2.1975];
ground_truth = importdata('/Users/aliceruget/Documents/PhD/Lindell/Test_HistSR_Net/Data/DATA_TEST_new/ground_truth.mat');
first_depth = importdata('/Users/aliceruget/Documents/PhD/Lindell/Alice_Training_dataset/Art_scene/depth_HR_before_down.mat');
first_depth_pp =importdata('/Users/aliceruget/Documents/PhD/Lindell/Test_HistSR_Net/Data/DATA_TEST_new/0_Df_down.mat');
resultat_hist = importdata('/Users/aliceruget/Documents/PhD/Lindell/Test_HistSR_Net/Data/DATA_TEST_new/resultat_in_meters.mat');
resultat_Lindell = importdata('/Users/aliceruget/Documents/PhD/Lindell/results/Art_2_50/resultat.mat');
ground_truth_bis = importdata('/Users/aliceruget/Documents/PhD/Lindell/results/Art_2_50/ground_truth.mat');
rmse_val_lindell = cal_rmse(resultat_Lindell,ground_truth_bis);


resultat_Lindell = resultat_Lindell(1:544, 1:688);

subplot(2,3,1)
imagesc(ground_truth)
%colorbar
caxis(axis_colorbar)
axis image
hold on 
rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
title({'\textbf{Ground Truth Depth}', '(a)'}, 'interpreter', 'latex')
axis off

% subplot(2,5,2)
% imagesc(first_depth)
% %colorbar
% caxis(axis_colorbar)
% axis image
% hold on 
% rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
% rmse_val = cal_rmse(first_depth,ground_truth)
% txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
% text(10,40,txt,'Color','black', 'Fontsize',12)%,'fontweight','bold')
% title({'\textbf{First Depth Map}', '(a)'}, 'interpreter', 'latex')
% axis off

% subplot(2,4,2)
% imagesc(first_depth_pp)
% %colorbar
% caxis([0,1])
% axis image
% hold on 
% rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
% rmse_val = cal_rmse(first_depth_pp,ground_truth)
% txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
% text(10,40,txt,'Color','black', 'Fontsize',12)%,'fontweight','bold')
% title({'\textbf{First Depth Map}','after cropping the histograms', '(b)'}, 'interpreter', 'latex')
% axis off

subplot(2,3,2)
imagesc(resultat_Lindell)
%colorbar
axis image
caxis(axis_colorbar)
txt = strcat('RMSE=',sprintf('%.3f',rmse_val_lindell));
text(10,40,txt,'Color','black', 'Fontsize',12)
hold on 
rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r','linewidth',1);
title({'\textbf{Lindell \textit{et al.} 2018}', 'on entire histogram','(b)'}, 'interpreter', 'latex')
axis off

subplot(2,3,3)
imagesc(resultat_hist)
%colorbar
axis image
rmse_val = cal_rmse(resultat_hist,ground_truth)
txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
text(10,40,txt,'Color','black', 'Fontsize',12)%,'fontweight','bold')

text(10,40,txt,'Color','black', 'Fontsize',12)

%text(10,70,txt,'Color','white', 'Fontsize',12,'fontweight','bold')
caxis(axis_colorbar)
axis off
hold on 
rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r','linewidth',1);
title({'\textbf{Proposed HistNet}','after cropping the histogram', '(c)'}, 'interpreter', 'latex')

subplot(2,3,4)
ground_truth_small = ground_truth(coord1,coord2);
imagesc(ground_truth_small)
caxis(axis_colorbar)
axis image
title('(a*)', 'interpreter', 'latex')
axis off
axis off

% subplot(2,4,7)
% first_depth_small = first_depth(coord1,coord2);
% imagesc(first_depth_small)
% caxis(axis_colorbar)
% axis image
% title('(d)', 'interpreter', 'latex')
% axis off
% axis off
% 
% subplot(2,4,6)
% first_depth_pp_small = first_depth_pp(coord1,coord2);
% imagesc(first_depth_pp_small)
% caxis([0,1])
% axis image
% title('(b*)', 'interpreter', 'latex')
% axis off
% axis off

subplot(2,3,5)
resultat_Lindell_small = resultat_Lindell(coord1,coord2);
imagesc(resultat_Lindell_small)
caxis(axis_colorbar)
axis image
title('(b*)', 'interpreter', 'latex')
axis off

subplot(2,3,6)
resultat_hist_small = resultat_hist(coord1,coord2);
imagesc(resultat_hist_small)
caxis(axis_colorbar)
axis image
title('(c*)', 'interpreter', 'latex')
axis off

text(31.8,281,'time = 7 s','Color','black', 'Fontsize',12, 'interpreter', 'latex')
text(-439,281,'time = 7 min','Color','black', 'Fontsize',12,'interpreter', 'latex')

%34.00869565217391,277.904347826087,0
cb = colorbar
caxis(axis_colorbar)
%cb.Ticks = [0,0.5,1];
%cb.TickLabels = {0,1,2};
set(get(cb,'label'),'string','Depth in m')
cb.Position = [0.911392405063291,0.104212860310421,0.008438818565398,0.798226164079823];

set(gcf, 'Position',[ 190   217   948   451])
%%
%% Compa Lindell with Confidence map
close all
coord1 = 100:350;
coord2 = 300:450;
axis_colorbar = [1.4357    2.1975];
ground_truth = importdata('/Users/aliceruget/Documents/PhD/Lindell/Test_HistSR_Net/Data/DATA_TEST_new/ground_truth.mat');
%first_depth = importdata('/Users/aliceruget/Documents/PhD/Lindell/Alice_Training_dataset/Art_scene/depth_HR_before_down.mat');
%first_depth_pp =importdata('/Users/aliceruget/Documents/PhD/Lindell/Test_HistSR_Net/Data/DATA_TEST_new/0_Df_down.mat');
%resultat_hist = importdata('/Users/aliceruget/Documents/PhD/Lindell/Test_HistSR_Net/Data/DATA_TEST_new/resultat_in_meters.mat');
first_depth = importdata('/Users/aliceruget/Documents/PhD/Confidence_map/Dataset/DATA_Lindell/0_Df_down.mat');
resultat_hist = importdata('/Users/aliceruget/Documents/PhD/Confidence_map/Dataset/DATA_Lindell/First_test_non_binary_3_460.mat');
resultat_Lindell = importdata('/Users/aliceruget/Documents/PhD/Lindell/results/Art_2_50/resultat.mat');
ground_truth_bis = importdata('/Users/aliceruget/Documents/PhD/Lindell/results/Art_2_50/ground_truth.mat');
rmse_val_lindell = cal_rmse(resultat_Lindell,ground_truth_bis);
confidence_map = importdata('/Users/aliceruget/Documents/PhD/Confidence_map/Dataset/DATA_Lindell/0_confidence.mat');

resultat_hist = (resultat_hist*53+98)*0.0293/2;
resultat_Lindell = resultat_Lindell(1:544, 1:688);

subplot(2,5,1)
imagesc(ground_truth)
%colorbar
caxis(axis_colorbar) 
axis image
hold on 
rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
title({'\textbf{Ground Truth Depth}', '(a)'}, 'interpreter', 'latex')
axis off

subplot(2,5,2)
imagesc(confidence_map)
cb = colorbar
cb.Position = [0.417721518987341,0.63563829787234,0.003164556962025,0.238758659079663];
%caxis(axis_colorbar)
axis image
hold on 
rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
title({'\textbf{Confidence map}', '(b)'}, 'interpreter', 'latex')
axis off


subplot(2,5,3)
imagesc(first_depth)
cb = colorbar;
%set(get(cb,'label'),'string','Depth in m')
cb.Position = [0.581223628691982,0.63563829787234,0.003164556962027,0.240892503711035];

%caxis(axis_colorbar)
axis image
hold on 
rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r','linewidth',1)
title({'\textbf{Fist depth input}', '(c)'}, 'interpreter', 'latex')
axis off


subplot(2,5,4)
imagesc(resultat_Lindell)
%colorbar
axis image
caxis(axis_colorbar)
txt = strcat('RMSE=',sprintf('%.3f',rmse_val_lindell));
text(10,40,txt,'Color','black', 'Fontsize',12)
hold on 
rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r','linewidth',1);
title({'\textbf{Proposed CNN of [2]}', 'on 544x688x1024 histogram','(d)'}, 'interpreter', 'latex')
axis off

subplot(2,5,5)
imagesc(resultat_hist)
%colorbar
axis image
rmse_val = cal_rmse(resultat_hist,ground_truth)
txt = strcat('RMSE=',sprintf('%.3f',rmse_val));
text(10,40,txt,'Color','black', 'Fontsize',12)%,'fontweight','bold')
%text(10,70,txt,'Color','white', 'Fontsize',12,'fontweight','bold')
caxis(axis_colorbar)
axis off
hold on 
rectangle('Position',[coord2(1),coord1(1), length(coord2), length(coord1)],'EdgeColor','r','linewidth',1);
title({'\textbf{Proposed HistNet}','on 136x172x1024 histogram','(4x downsampled)', '(e)'}, 'interpreter', 'latex')

subplot(2,5,6)
ground_truth_small = ground_truth(coord1,coord2);
imagesc(ground_truth_small)
caxis(axis_colorbar)
axis image
title('(a*)', 'interpreter', 'latex')
axis off
axis off

subplot(2,5,7)
confidence_map_small = confidence_map(coord1,coord2);
imagesc(confidence_map_small)
colorbar
%caxis(axis_colorbar)
axis image
title({'(b*)'}, 'interpreter', 'latex')
axis off

subplot(2,5,8)
first_depth_small = first_depth(coord1,coord2);
imagesc(first_depth_small)
colorbar
%caxis(axis_colorbar)
axis image
title({'(c*)'}, 'interpreter', 'latex')
axis off

subplot(2,5,9)
resultat_Lindell_small = resultat_Lindell(coord1,coord2);
imagesc(resultat_Lindell_small)
caxis(axis_colorbar)
axis image
title('(d*)', 'interpreter', 'latex')
axis off

subplot(2,5,10)
resultat_hist_small = resultat_hist(coord1,coord2);
imagesc(resultat_hist_small)
caxis(axis_colorbar)
axis image
title('(e*)', 'interpreter', 'latex')
axis off

cb = colorbar
caxis(axis_colorbar)
%cb.Ticks = [0,0.5,1];
%cb.TickLabels = {0,1,2};
set(get(cb,'label'),'string','Depth in m')
cb.Position = [0.914563106796117,0.109042553191489,0.005268116832575,0.76339480443949];

set(gcf, 'Position',[ 190   217   948   376])

%%
close all
imagesc(intensity)
axis image
axis off
colormap(gray)

%% RMSE
close all;
tab = importdata('/Users/aliceruget/Documents/PhD/Two-Depth_SR_Net/repository_paper/simulated_data/multiple_level/rmse_tab.mat')

tab = cell2mat(tab);
plot(tab(2:end,1))
xlabel('value of level to define second depth','interpreter','latex')

%xticklabels({'x = 0','x = 5','x = 10'})
ylabel('RMSE value','interpreter','latex')

function[rmse_value] = cal_rmse( image1, image2)
    dif = (image1 - image2).^2;
    dif_sum = mean(dif, 'all');
    rmse_value = sqrt(dif_sum);
end
function [absolute_value] = cal_ae(image1,image2)
    dif = abs(image1 - image2);
    absolute_value = mean(dif, 'all');    
end

function axesoffwithlabels(h)
%AXESOFFWITHLABELS Make axes invisible but not the xlabel and ylabel.
%
%   AXESOFFWITHLABELS(H) makes axes invisible, keeping the x- and ylabel
%   with handle H.
%
%  Sample Usage
%    plot(rand(1,10))
%    h(1) = xlabel('x');
%    h(2) = ylabel('x');
%    axesoffwithlabels(h)
%
%   Thorsten.Hansen@psychol.uni-giessen.de  2018-08-08
set(gca, 'Xcolor', 'w', 'Ycolor', 'w')
set(h, 'Color', 'k')
% get rid of the white ticks and tick labels, moving the labels closer to
% the axes
set(gca, 'XTick', []);
set(gca, 'YTick', []);
end

function plot_3d_figure(recon)
DepthImage = squeeze(recon);
r1 = size(DepthImage,1);
c1 = size(DepthImage,2);
DProc=reshape(DepthImage,r1,c1);
scatter3(kron(ones(r1,1),(1:c1)'),reshape(DProc(end:-1:1,:)',r1*c1,1)',kron((1:r1)',ones(c1,1)),80,reshape(DProc(end:-1:1,:)',r1*c1,1)','.'),
%colorbar
axis off
end