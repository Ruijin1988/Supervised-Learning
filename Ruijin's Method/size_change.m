for ii = 1:60
    temp=reshape(xtr(ii,:),[82944/64 64]);
    for i = 1:64
        temp2=reshape(temp(:,i),[36 36]);
        temp2=imresize(temp2,[18 18]);
        xtr_sm(:,i)=temp2(:);
    end
    xtr_sm_store(ii,:)=im2bw(xtr_sm(:));
end
clear temp
clear temp2

for jj = 1:60
    temp=reshape(xtr_sm_store(jj,:),[9*9*4*64/64 64]);
    for j = 1:64
        temp2=reshape(temp(:,j),[18 18]);
        temp2=imresize(temp2,[36 36]);
        xtr_lg(:,j)=temp2(:);
    end
    xtr_lg_store(jj,:)=xtr_lg(:);
end