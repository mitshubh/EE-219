Indices = crossvalind('Kfold', 100000, 10);
predicted = NaN(943,1682,3);
i = 1;

K = [10, 50, 100];
file = 'u.data';
delimiter = ('\t');
u= dlmread(file, delimiter);

for kloop=1:3

    for i=1:10
    Test = zeros(10000,5);
    RTrain = NaN(943,1682);
    WTrain = NaN(943,1682);
    k = 1;
    for j=1:100000
        if(Indices(j) ~= i)
            WTrain(u(j,1),u(j,2)) = u(j,3); 
            RTrain(u(j,1),u(j,2)) = 1;
        else
            Test(k,1) = u(j,1);	
            Test(k,2) = u(j,2); 
            Test(k,3) = u(j,3); 
            k=k+1;
        end
    end
    [U,V] = wnmfrule(RTrain,K(kloop)); 
    UV = U*V;
    
    for j=1:10000
        Test(j,4) = UV(Test(j,1),Test(j,2)); 
        Test(j,5) = abs(Test(j,3) - Test(j,4)); 
        predicted(Test(j,1),Test(j,2),kloop) = UV(Test(j,1),Test(j,2)); 
    end
    
    end
    
end




hitRate = zeros(943,20,3); 
faRate = zeros(943,20,3); 
thresh = 0.4; 
precision = zeros(943,3); 


for m=1:3 
    for i=1:943 
    
    
    [~, indices] = sort(predicted(i,:,m),'descend');
    
    
    for L=1:20
        top_movies = zeros(943,L,3); 
        count = 1; 
       
        for t=1:size(indices,2)
            
            if RTrain(i,indices(t))== 1
                top_movies(i,count,m) = indices(t);
                count = count+1;
            end
            
           
            if count == L+1
                
                
                tp = length(find((predicted(i,top_movies(i,:,m),m)> thresh) & (WTrain(i,top_movies(i,:,m))>3)));
                tn = length(find((predicted(i,top_movies(i,:,m),m)<= thresh) & (WTrain(i,top_movies(i,:,m))<=3)));
                fp = length(find((predicted(i,top_movies(i,:,m),m)> thresh) & (WTrain(i,top_movies(i,:,m))<=3)));
                fn = length(find((predicted(i,top_movies(i,:,m),m)<= thresh) & (WTrain(i,top_movies(i,:,m))>3)));

                
                if L==5
                    precision(i,m) = tp/length(find(predicted(i, top_movies(i,:,m),m)> thresh));
                end
                
                
                if tp==0 && fn==0
                   hitRate(i,L,m) = 0; 
                else
                    hitRate(i,L,m) = tp/(tp+fn);
                end
                
                
                if fp==0 && tn==0
                    faRate(i,L,m) = 0; 
                else
                    faRate(i,L,m) = fp/(fp+tn);
                end
                
                break;    
            
            end
        end
    end
    end 
end

meanHr = zeros(20,3);
meanFa = zeros(20,3);

avgPrecision = zeros(3);
for m=1:3
    avgPrecision(m) =  mean(precision(:,m));
end

for m=1:3
    for L=1:20
        meanHr(L,m) = mean(hitRate(:,L,m));
        meanFa(L,m) = mean(faRate(:,L,m));
    end
end

 plot((1:20),meanHr(:,1),'r',(1:20),meanHr(:,2),'b',(1:20),meanHr(:,3),'g')
 xlabel('L')
 ylabel('Average Hit Rate')
 legend('k = 10', 'k = 50', 'k = 100')

 plot((1:20),meanFa(:,1),'r',(1:20),meanFa(:,2),'b',(1:20),meanFa(:,3),'g')
 xlabel('L')
 ylabel('Average False Alarm Rate')
 legend('k = 10', 'k = 50', 'k = 100')

plot(meanFa(:,1),meanHr(:,1),'r',meanFa(:,2),meanHr(:,2),'b',meanFa(:,3),meanHr(:,3),'g')
 xlabel('Average False Alarm Rate')
 ylabel('Average Hit Rate')
 legend('k = 10', 'k = 50', 'k = 100')
