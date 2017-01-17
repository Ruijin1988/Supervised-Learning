% function [Y_data_model]=Y_data(image_temp,channel_temp,j,k)
%     for neighbor_x = 1:sqrt(size(image_temp,1))-5
%         for neighbor_y = 1:sqrt(size(image_temp,1))-7
%             Y_data_model(k,1)=channel_temp{j}(neighbor_x+5,neighbor_y+4);
%             k = k+1;
%         end
%     end
% end

function [Y_data_model]=Y_data(image_temp,channel_temp,j,k)
    for neighbor_x = 1:sqrt(size(image_temp,1))-2
        for neighbor_y = 1:sqrt(size(image_temp,1))-4
            Y_data_model(k,1)=channel_temp{j}(neighbor_x+2,neighbor_y+2);
            k = k+1;
        end
    end
end