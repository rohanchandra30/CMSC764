function count = numsamples(labels, output)
count = 0;
for i = 1:length(labels)
    if labels(i) == output(i)
        count = count + 1;
    end
end
end