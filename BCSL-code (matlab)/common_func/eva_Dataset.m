function [F1,Precision,Recall,Distance]=eva_Dataset(learned_PCs,true_PCs)

f1s=[];
precisions=[];
recalls=[];
distances=[];

if size(learned_PCs,1)==1
    for i=1:size(learned_PCs,2)
        [f1,precision,recall,distance]=eva_PC(learned_PCs{i},true_PCs{i});
        f1s=[f1s f1];
        precisions=[precisions precision];
        recalls=[recalls recall];
        distances=[distances distance];
    end
else
    for i=1:size(learned_PCs,2)
        tmp_PCs=[];
        for j=1:size(learned_PCs,2)
            if learned_PCs(i,j)==1
                tmp_PCs=[tmp_PCs j];
            end
        end
  
        [f1,precision,recall,distance]=eva_PC(tmp_PCs,true_PCs{i});
        f1s=[f1s f1];
        precisions=[precisions precision];
        recalls=[recalls recall];
        distances=[distances distance];
    end
end

F1=mean(f1s);
Precision=mean(precisions);
Recall=mean(recalls);
Distance=mean(distances);

end