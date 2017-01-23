function circle(x,y,r)
ang=0:0.01:2*pi; 
xp=r*cos(ang);
yp=r*sin(ang);
axis equal
xlim([0 200])
ylim([0 200])
axis off
plot(x+xp,y+yp,'k');
end