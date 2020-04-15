% Sena Nur
% Bİl.MÜH.(YL)
function som = som_proje(nfeatures, ndim, nepochs, negitimvectors, eta0, etadecay, sgm0, sgmdecay, showMode)
      
%       som = som_proje(2,10,5,10,0.1,0.05,20,0.05,2);
%       som = som_proje(2,10,10,100,0.1,0.05,20,0.05,2);

%       som = som_proje(3,100,5,100,0.1,0.05,20,0.05,2);

%       % Kullanýlanlar:
%       %   3   : RGB deðerleri gibi eðitim vektörleri için boyutlar
%       %   60x60: nöronlar
%       %   10   : epochlar
%       %   100  : egitilen vektör
%       %   0.1  : ilk ögrenme oraný
%       %   0.05 : öðrenme oranlarýnýn üstel bozulma oraný
%       %   20   : ilk gauss oraný
%       %   0.05 :Gauss varyansýnýn üstel bozulma oraný
%       %   2    : Güncellemeden sonra haritayý göster

% RGB degerleri için (3) kullanmamýz gerekiyor.
% Fakat vektörü küçük kullandýgým için þimdilik etkisi yok büyük
% vektörlerle çok daha güzel sonuçlar çýkmaktadýr.
% Deðerleri Deðiþitirmek Kullanýcýya aitttir.




nrows = ndim;
ncols = ndim;
som = rand(nrows,ncols,nfeatures);
if showMode >= 1
    fig = figure;
    gosterSOMmap(fig, 1, 'Random Þekilde Oluþturulan SOM haritasý ', som, nfeatures);
end
% Rastgele eðitim verisi oluþtur
egitimData = rand(negitimvectors,nfeatures);
% Koordinat Sistemi Oluþturma
[x y] = meshgrid(1:ncols,1:nrows);
for t = 1:nepochs   
    % Mevcut dönem için öðrenme oranýný epoch hesaplama
    eta = eta0 * exp(-t*etadecay);        
   % Geçerli dönem için Gaussian (Neighborhood-komþuluk) fonksiyonunun varyansýný hesaplayýn
    sgm = sgm0 * exp(-t*sgmdecay);
    
    %Gauss fonksiyonunun geniþliðini 3 sigma olarak düþünün.
    width = ceil(sgm*3);        
    %% %%%%%%%%%%%%%%%%%%
    
    %Bu Bölümde Yardým Aldým
    
    for ntraining = 1:negitimvectors
        % eðitim vektörü oluþturma
        trainingVector = egitimData(ntraining,:);
                
        % Eðitim vektörü ile Öklid mesafesini hesaplayýn
        
        % SOM haritasýnda her bir nöronun yüzdesi
        dist = getEuclideanDistance(trainingVector, som, nrows, ncols, nfeatures);
        
       %En iyi eþleþmeyi bul bmu için
        [~, bmuindex] = min(dist); 
        
        %bmu indexlerini 2D ye dönüþtür
        [bmurow bmucol] = ind2sub([nrows ncols],bmuindex);        
                
      % Bmu'nun bulunduðu yere odaklanan bir Gauss iþlevi oluþturun
        g = exp(-(((x - bmucol).^2) + ((y - bmurow).^2)) / (2*sgm*sgm));
                        
       % Yakýn komþularýn sýnýrýný belirleme
        fromrow = max(1,bmurow - width);
        torow   = min(bmurow + width,nrows);
        fromcol = max(1,bmucol - width);
        tocol   = min(bmucol + width,ncols);
        
       %Komþu nöronlarý alýn ve komþu boyutunu belirlemek
        neighbourNeurons = som(fromrow:torow,fromcol:tocol,:);
        sz = size(neighbourNeurons);
        
        % Eðitim vektörünü ve Gauss fonksiyonunu dönüþtürün
        %Nöron aðýrlýklarý güncellemesinin hesaplanmasýný kolaylaþtýrmak için
        % çok boyutlu yapma
        
        T = reshape(repmat(trainingVector,sz(1)*sz(2),1),sz(1),sz(2),nfeatures);                   
        G = repmat(g(fromrow:torow,fromcol:tocol),[1 1 nfeatures]);
        
        % Bmu bölgesinde bulunan nöronlarýn aðýrlýklarýný güncelle
        neighbourNeurons = neighbourNeurons + eta .* G .* (T - neighbourNeurons);
        
       %BMU komþu nöronlarýnýn yeni aðýrlýklarýný SOM haritasýnýn tamamýna geri koyun
        som(fromrow:torow,fromcol:tocol,:) = neighbourNeurons;
        if showMode == 2
            gosterSOMmap(fig, 2, ['Epoch: ',num2str(t),'/',num2str(nepochs),', Eðitilmiþ Vektör: ',num2str(ntraining),'/',num2str(negitimvectors)], som, nfeatures);
        end        
    end
end
if showMode == 1
    gosterSOMmap(fig, 2, 'Eðitilmiþ SOM Haritasý', som, nfeatures);
end
function ed = getEuclideanDistance(egitimVector, sommap, nrows, ncols, nfeatures)
% Nöronlarýn 3B gösterimini 2B'ye dönüþtürün
neuronList = reshape(sommap,nrows*ncols,nfeatures);               
% Öklid Mesafesini Sýfýrla
ed = 0;
for n = 1:size(neuronList,2)
    ed = ed + (egitimVector(n)-neuronList(:,n)).^2;
end
ed = sqrt(ed);
function gosterSOMmap(fig, nsubplot, description, sommap, nfeatures)
% Verilen SOM haritasýný görüntüle

figure(fig);
subplot(1,2,nsubplot);
if nfeatures >= 3
    imagesc(sommap(:,:,1:3)); 
    %Alternatif olarak, resim kullanmak yerine deðerleri ölçeklendirmek 
    %için imagesc iþlevini kullanabilirsiniz 
else
    imagesc(sommap(:,:,1));
end

title(description);
