clear
for ii = 1:60
r=3;k=1;mid=[randi([41,159],1),randi([41,159],1)];R=40;
[rr cc] = meshgrid(1:200);
circle=zeros(200,200);
for i = 1:200
    x_coor=randi([r+1,200-r-1],1);
    y_coor=randi([r+1,200-r-1],1);
    dist=sqrt((x_coor-mid(1))^2+(y_coor-mid(2))^2);
    if dist>R
        continue
    end
    coor{k}=[x_coor,y_coor];
    coor_cur=coor{k};
    diff=[];
    for j = 1:k
        diff=[diff,sqrt((coor_cur-coor{j})*(coor_cur-coor{j})')];
    end
    if numel(find(diff<=2*r))>1
        continue
    end     
    C = sqrt((rr-x_coor).^2+(cc-y_coor).^2)<=r;
    circle = circle + double(C);
%     circle(x_coor,y_coor,r);
%     hold on;
    k=k+1;
end
circle_store(ii,:)=circle(:);
end