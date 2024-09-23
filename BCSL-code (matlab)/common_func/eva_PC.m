function [F1,precision,recall,distance]=eva_PC(PC,truePC)

if isempty(truePC)
    if isempty(PC)
        precision=1;
        recall=1;
        distance=0;
        F1=1;
    elseif ~isempty(PC)
        precision=0;
        recall=0;
        distance=sqrt(2);
        F1=0;
    end
elseif ~isempty(truePC)
    if ~isempty(PC)
        precision_tmp=length(intersect(PC,truePC))/length(PC);
        recall_tmp=length(intersect(PC,truePC))/length(truePC);
        distance_tmp=sqrt((1-recall_tmp)*(1-recall_tmp)+(1-precision_tmp)*(1-precision_tmp));
        if (precision_tmp+recall_tmp)==0
            f1_tmp=0;
        else
            f1_tmp=2*precision_tmp*recall_tmp/(precision_tmp+recall_tmp);
        end
        precision=precision_tmp;
        recall=recall_tmp;
        distance=distance_tmp;
        F1=f1_tmp;
    else
        precision=0;
        recall=0;
        distance=sqrt(2);
        F1=0;
    end
end

