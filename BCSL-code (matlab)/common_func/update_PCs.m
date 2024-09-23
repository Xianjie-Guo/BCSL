function [PCs_new] = update_PCs(conflicts_node_pairs,PCs,Score)

for i=1:length(Score)
    if Score(i)>0 
        PCs{conflicts_node_pairs{i}(1)}=myunion(PCs{conflicts_node_pairs{i}(1)}, conflicts_node_pairs{i}(2));
        PCs{conflicts_node_pairs{i}(2)}=myunion(PCs{conflicts_node_pairs{i}(2)}, conflicts_node_pairs{i}(1));
    else 
        PCs{conflicts_node_pairs{i}(1)}=mysetdiff(PCs{conflicts_node_pairs{i}(1)}, conflicts_node_pairs{i}(2));
        PCs{conflicts_node_pairs{i}(2)}=mysetdiff(PCs{conflicts_node_pairs{i}(2)}, conflicts_node_pairs{i}(1));
    end
end

PCs_new=PCs;

end

