function [Data_Bootstrap_N]=gen_bootstrap_data(Data,rand_sample_numb)

[q,~]=size(Data);

Data_Bootstrap_N=cell(1,rand_sample_numb);

for i=1:rand_sample_numb
    index=ceil(rand(1,q)*q);
    index=index';
    Data_Bootstrap_N{i} = Data(index,:);
end

end