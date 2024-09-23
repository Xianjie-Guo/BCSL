function [DAG, time] = BCSL(Data, Alpha, rand_sample_numb)
% by Xianjie Guo 2022.11.24

% input:
%     Data: the values of each variable in the data matrix start from 1.
%     Alpha: significance level, e.g., 0.01 or 0.05.
%     rand_sample_numb: the number of sub-datasets generated.
% output:
%     DAG: a directed acyclic graph learned on a given dataset.
%     time: the runtime of the algorithm.

ns=max(Data);
[q,p]=size(Data);
maxK=3;

start=tic;

varNValues=ns;
sample=Data;

PCs=cell(1,p);
skeleton=zeros(p,p);
for i=1:p
    [pc,~,~,~]=HITONPC_G2_CI0(sample,i,Alpha,ns,p,maxK);
    PCs{i}=pc;
    skeleton(i,pc)=1;
end

conflicts_node_pairs={};
for i=1:p
    for j=1:i-1
        if skeleton(i,j)~=skeleton(j,i)
            conflicts_node_pairs{end+1}=[i,j];
        end
    end
end

results_D_n=cell(2,length(conflicts_node_pairs));
weights_D_n=cell(2,length(conflicts_node_pairs));
for i=1:rand_sample_numb
    index=ceil(rand(1,q)*q);
    index=index';
    Data_Bootstrap = Data(index,:);
    
    ns=max(Data_Bootstrap);
    
    for j=1:length(conflicts_node_pairs)
        [pc1,~,~,~]=HITONPC_G2_CI1(Data_Bootstrap,conflicts_node_pairs{j}(1),Alpha,ns,p,maxK);
        [F1,~,~,~]=eva_PC(pc1,PCs{conflicts_node_pairs{j}(1)});
        weights_D_n{1,j}=[weights_D_n{1,j} F1];
        if ismember(conflicts_node_pairs{j}(2),pc1)
            results_D_n{1,j}=[results_D_n{1,j} 1];
        else
            results_D_n{1,j}=[results_D_n{1,j} -1];
        end
        
        [pc2,~,~,~]=HITONPC_G2_CI1(Data_Bootstrap,conflicts_node_pairs{j}(2),Alpha,ns,p,maxK);
        [F1,~,~,~]=eva_PC(pc2,PCs{conflicts_node_pairs{j}(2)});
        weights_D_n{2,j}=[weights_D_n{2,j} F1];
        if ismember(conflicts_node_pairs{j}(1),pc2)
            results_D_n{2,j}=[results_D_n{2,j} 1];
        else
            results_D_n{2,j}=[results_D_n{2,j} -1];
        end
    end
end

Score=zeros(1,length(conflicts_node_pairs))*(-100);
for i=1:length(conflicts_node_pairs)
    Score(i)=(sum(results_D_n{1,i}.*weights_D_n{1,i})+sum(results_D_n{2,i}.*weights_D_n{2,i}))/(2*rand_sample_numb);
end

for i=1:length(Score)
    if Score(i)>0
        skeleton(conflicts_node_pairs{i}(1),conflicts_node_pairs{i}(2))=1;
        skeleton(conflicts_node_pairs{i}(2),conflicts_node_pairs{i}(1))=1;
    else
        skeleton(conflicts_node_pairs{i}(1),conflicts_node_pairs{i}(2))=0;
        skeleton(conflicts_node_pairs{i}(2),conflicts_node_pairs{i}(1))=0;
    end
end

cpm = tril(sparse(skeleton));

% create local scorer
LocalScorer = bdeulocalscorer(sample, varNValues);

% create hill climber
HillClimber = hillclimber(LocalScorer, 'CandidateParentMatrix', cpm);

% Finally, we learn the structure of the network.
% learn structure
DAG = HillClimber.learnstructure();
DAG=full(DAG);

time=toc(start);

end