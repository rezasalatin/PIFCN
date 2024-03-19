%close(1); close(2);
clear 
path = 'lu000_lr001_bs2_y1_PRELU_x1/';
pred = load([path 'prediction_epoch_1000.mat']);
input = load('../dataset/input_data.mat');
target = load('../dataset/target_data.mat');

x_int = 4; y_int = 64;

X = squeeze(input.input_data(1,1,:,:));
Y = squeeze(input.input_data(1,2,:,:));
h = squeeze(target.h(1,1,:,:));
pred = squeeze(pred.prediction(1,1,:,:));

figure(1)
plot(X(1,:),-h(129,:),'k','linewidth',2)
hold on
plot(X(1,:),-pred(129,:),'linewidth',2)
plot(X(1:y_int:end,1:x_int:end),-h(1:y_int:end,1:x_int:end),'.k','MarkerSize',25)
xlim([25.8 32.1])
title('mid LS')
print('-dpng', '-r300', [path 'mid LS'])


figure(2)
plot(X(1,:),-h(193,:),'k','linewidth',2)
hold on
plot(X(1,:),-pred(193,:),'linewidth',2)
xlim([25.8 32.1])
title('upper LS')
print('-dpng', '-r300', [path 'upper LS'])


figure(3)
plot(X(1,:),-h(65,:),'k','linewidth',2)
hold on
plot(X(1,:),-pred(65,:),'linewidth',2)
xlim([25.8 32.1])
title('lower LS')
print('-dpng', '-r300', [path 'lower LS'])


figure()
pcolor(X,Y,-pred(:,:)); shading interp; 
hold on
plot(X(1:y_int:end,1:x_int:end),Y(1:y_int:end,1:x_int:end),'.k','MarkerSize',15)
plot(X(end,1:x_int:end),Y(end,1:x_int:end),'.k','MarkerSize',15)
plot(X(1:y_int:end,end),Y(1:y_int:end,end),'.k','MarkerSize',15)
plot(X(end,end),Y(end,end),'.k','MarkerSize',15)
colorbar
colormap hot
clim([-0.6 0])
print('-dpng', '-r300', [path 'Pred'])

