function y = sem(x,dim)

if (nargin == 1)
	dim = min(find(size(x)~=1));
	if isempty(dim)
		dim = 1;
	end
end

y = sqrt(var(x,0,dim)/size(x,dim));
