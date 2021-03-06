clear
load('alloy2_neighbor.mat');
load('alloy2_label.mat');
window_size=12;
neighbor_target=[];
label_target=[];
for i = 1:1
    neighbor_temp=neighbor{i};
    label_temp=Y{i};
    neighbor_target=[neighbor_target,neighbor_temp'];
    label_target=[label_target, label_temp'];
end
X=neighbor_target';
Y=label_target';
ctree = fitctree(X,Y);

x_inital=randi(100,[200,200]);
x_inital(x_inital>=67)=0;
x_inital(x_inital~=0)=1;
load('alloy2.mat');
x_inital=reshape(xtr(1,:),[200 200]);
x_original=reshape(xtr(1,:),[200 200]);
for i = 1:100
    fprintf('prediction loop %d...\n',i);
    for neighbor_x = 1:size(x_inital,1)-2
        for neighbor_y = 1:size(x_inital,1)-4
            X_data_temp1=x_inital(neighbor_x:neighbor_x+1, neighbor_y:neighbor_y+4);
            X_data_temp2=x_inital(neighbor_x+2,neighbor_y:neighbor_y+1);
            X_data_temp1 = X_data_temp1';
            X_data_temp2 = X_data_temp2';
            X_data_temp = [X_data_temp1(:)',X_data_temp2(:)'];
            [Y_temp,score ]= predict(ctree,X_data_temp);
            if score(2)>rand(1)
                Y_temp=1;
            else
                Y_temp=0;
            end
            x_inital(neighbor_x+2,neighbor_y+2)=Y_temp;
        end
    end
    diff=sum(x_inital(:)-x_original(:))^2/numel(x_inital)
end