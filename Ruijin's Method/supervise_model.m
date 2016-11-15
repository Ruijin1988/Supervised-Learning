clear;
xtr=load('3rdlayer_data.mat');
xtr=xtr.xtr;
k=1;image_num=2;channel=288; total_model=287;
%% extract the neighbor value and their related pixel value
image_num=100;channel=288;
for i = 1:image_num
    fprintf('Loading image %d...\n',i);
    image_temp=reshape(xtr(i,:),[size(xtr,2)/channel, channel]);
    for j = 1:channel
    	channel_temp{j}=reshape(image_temp(:,j),[sqrt(size(image_temp,1)),sqrt(size(image_temp,1))]);
        neighbor{j+(i-1)*channel}=neighbor_data(image_temp, channel_temp,j,k);
        Y{j+(i-1)*channel}=Y_data(image_temp, channel_temp,j,k);
    end
        
end
%% To calculate the 287 supervised causal model
for tm = 1:total_model
    fprintf('perparing the data for model_%d...\n',tm+1);
    X_input{tm} = [];Y_output{tm}=[];
    for i = 1:image_num
        X_input_temp=[];
        for j = 1:tm
            X_input_temp = [X_input_temp, neighbor{j+(i-1)*channel}];
        end
        X_input{tm} = [X_input{tm},X_input_temp'];
        Y_output{tm}= [Y_output{tm},Y{tm+1+(i-1)*channel}'];
    end
end