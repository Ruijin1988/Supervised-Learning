clear;
xtr=load('alloy2.mat');
xtr=xtr.xtr;
k=1;
%% extract the neighbor value and their related pixel value
image_num=1;channel=1;
for i = 1:image_num
    fprintf('Loading image %d...\n',i);
    image_temp=reshape(xtr(i,:),[size(xtr,2)/channel, channel]);
    for j = 1:channel
    	channel_temp{j}=reshape(image_temp(:,j),[sqrt(size(image_temp,1)),sqrt(size(image_temp,1))]);
        neighbor{j+(i-1)*channel}=neighbor_data(image_temp, channel_temp,j,k);
        Y{j+(i-1)*channel}=Y_data(image_temp, channel_temp,j,k);
    end        
end