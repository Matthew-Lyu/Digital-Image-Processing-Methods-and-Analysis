function count = Verticalprojection(candidate)
%candidate is a rectangle 
%it's gray img's area
    candidate= imbinarize(candidate,0.5);
   
    [m,n]=size(candidate);

    for yi=1:n
        Si(yi)=sum(candidate(:,yi));
    end
    %求投影灰度函数反复穿过他平均值的次数
    mean_num=0;
    for yi=1:n
        mean_num=mean_num+Si(yi);
    end
    mean_num=mean_num/n;
    
    for yi=1:n
        Si(yi)=Si(yi)-mean_num;
    end
    
    count=0;
    for yi=1:n-1
        if Si(yi)*Si(yi+1)<0
            count=count+1;
        end
    end
end