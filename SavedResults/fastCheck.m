%close(1); close(2);
clear 
path = 'lu010_lr001_bs2_y32_x4_PRELU_pretrained_best/';
pred = load([path 'predictions_epoch_400.mat']);
input = load('../dataset/train_input.mat');
target = load('../dataset/train_target.mat');

x_int = 4; y_int = 32;

X = squeeze(input.train_input(1,1,:,:));
Y = squeeze(input.train_input(1,2,:,:));
h = squeeze(target.train_target(1,1,:,:));
pred = squeeze(pred.prediction(1,1,:,:));

figure(1)
plot(X(1,:),-h(129,:),'k','linewidth',2)
hold on
plot(X(1,:),-pred(129,:),'linewidth',2)
plot(X(1:y_int:end,1:x_int:end),-h(1:y_int:end,1:x_int:end),'.k','MarkerSize',15)
xlim([25.8 32.1])
xlabel('X (m)'); ylabel('Elev (m)');
set(gca,'FontSize',15)
print('-dpng', '-r300', [path 'mid LS'])


figure(2)
plot(X(1,:),-h(193,:),'k','linewidth',2)
hold on
plot(X(1,:),-pred(193,:),'linewidth',2)
xlim([25.8 32.1])
xlabel('X (m)'); ylabel('Elev (m)');
set(gca,'FontSize',15)
print('-dpng', '-r300', [path 'upper LS'])


figure(3)
plot(X(1,:),-h(65,:),'k','linewidth',2)
hold on
plot(X(1,:),-pred(65,:),'linewidth',2)
xlim([25.8 32.1])
xlabel('X (m)'); ylabel('Elev (m)');
set(gca,'FontSize',15)
print('-dpng', '-r300', [path 'lower LS'])


figure()
pcolor(X,Y,-pred(:,:)); shading interp; 
hold on
plot(X(1:y_int:end,1:x_int:end),Y(1:y_int:end,1:x_int:end),'.k','MarkerSize',8)
plot(X(end,1:x_int:end),Y(end,1:x_int:end),'.k','MarkerSize',8)
plot(X(1:y_int:end,end),Y(1:y_int:end,end),'.k','MarkerSize',8)
plot(X(end,end),Y(end,end),'.k','MarkerSize',8)
colorbar
colormap hot
clim([-0.6 0])
xlabel('X (m)'); ylabel('Y (m)');
set(gca,'FontSize',15)
print('-dpng', '-r300', [path 'Pred'])

