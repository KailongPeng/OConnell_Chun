% function memcheck
memout=whos;
fprintf('%2.4f gigabytes\n',(sum([memout.bytes]) / 1024 / 1024 / 1024));

%MONITOR_MEMORY_WHOS uses the WHOS command and evaluates inside the BASE
%workspace and sums up the bytes.  The output is displayed in MB.

% mem_elements = evalin('base','whos');
% if size(mem_elements,1) > 0
% 
%     for i = 1:size(mem_elements,1)
%         memory_array(i) = mem_elements(i).bytes;
%     end
% 
%     memory_in_use = sum(memory_array);
%     memory_in_use = memory_in_use/1024/1024/1024;
% else
%     memory_in_use = 0;
% end
% fprintf('%2.4f gigabytes\n',memory_in_use);