N = 10000000;

compare = false;
if (compare)
    py_list = 1.7;
    py_numpy = 28.3/1000;
end

v = ones(1,N);

% Using a loop
tic
for i = 1:N
    w(i) = exp(v(i));
end
t = toc;
fprintf('%30s %10g ms\n','Elapsed time (loop)',1000*t);
if (compare)
    fprintf('%30s %10g\n','Speedup over list',py_list/t);
    fprintf('%30s %10g\n','Speedup over numpy',py_numpy/t);
end
fprintf('\n');

% Vectorized
clear w
tic
w = exp(v);
t = toc;
fprintf('%30s %10g ms\n','Elapsed time (vectorized)',1000*t);
if (compare)
    fprintf('%30s %10g\n','Speedup over list',py_list/t);
    fprintf('%30s %10g\n','Speedup over numpy',py_numpy/t);
end
fprintf('\n');
