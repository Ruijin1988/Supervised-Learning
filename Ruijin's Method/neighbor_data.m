% function [X_data_model]=neighbor_data(image_temp,channel_temp,j,k)
%     for neighbor_x = 1:sqrt(size(image_temp,1))-5
%         for neighbor_y = 1:sqrt(size(image_temp,1))-7
%             X_data_temp1=channel_temp{j}(neighbor_x:neighbor_x+4, neighbor_y:neighbor_y+7);
%             X_data_temp2=channel_temp{j}(neighbor_x+5,neighbor_y:neighbor_y+3);
%             X_data_temp1 = X_data_temp1';
%             X_data_temp2 = X_data_temp2';
%             X_data_temp = [X_data_temp1(:)',X_data_temp2(:)'];
%             X_data_model(k,:) = X_data_temp;
% %             Y_data_model(k,1)=channel_temp2(neighbor_x+2,neighbor_y+2);
%             k = k+1;
%         end
%     end
% end

function [X_data_model]=neighbor_data(image_temp,channel_temp,j,k)
    for neighbor_x = 1:sqrt(size(image_temp,1))-2
        for neighbor_y = 1:sqrt(size(image_temp,1))-4
            X_data_temp1=channel_temp{j}(neighbor_x:neighbor_x+1, neighbor_y:neighbor_y+4);
            X_data_temp2=channel_temp{j}(neighbor_x+2,neighbor_y:neighbor_y+1);
            X_data_temp1 = X_data_temp1';
            X_data_temp2 = X_data_temp2';
            X_data_temp = [X_data_temp1(:)',X_data_temp2(:)'];
            X_data_model(k,:) = X_data_temp;
%             Y_data_model(k,1)=channel_temp2(neighbor_x+2,neighbor_y+2);
            k = k+1;
        end
    end
end