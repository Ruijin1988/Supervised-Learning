%% calculate the mean-similarity
error_total = zeros(288,288);
for i = 1:60
    fprintf('Loading image %d...\n',i);
    xtr_temp=reshape(xtr(i,:),[1296 288]);
    for j = 1:288
        for k = 1:288
            error_temp=abs(xtr_temp(:,j)-xtr_temp(:,k)).^2/numel(xtr_temp(:,j));
            error(j,k)=sum(error_temp(:));
        end
    end
    error_store{i}=error;
    error_total=error+error_total;
end
error_mean=error_total/60;

%% do clustering
U=triu(error_mean);
dic_vect=[];
for i = 1:288
    for j = 1:288
        if U(i,j)~=0
            dic_vect=[dic_vect, error_mean(i,j)];
        else
            continue
        end
    end
end
Z = linkage(dic_vect);
T = cluster(Z,'maxclust',5);

variance_store=[];
average_store=[];
for kk = 1:60
    fprintf('Comparing image %d...\n',kk);
    error=error_store{kk};
    U=triu(error);
    dic_vect=[];
    for i = 1:288
        for j = 1:288
            if U(i,j)~=0
                dic_vect=[dic_vect, error(i,j)];
            else
                continue
            end
        end
    end
%     Z = linkage(dic_vect);
%     T{kk} = cluster(Z,'maxclust',20);
    variance_store=[variance_store,var(dic_vect)];
    average_store=[average_store, mean(dic_vect)];
end


