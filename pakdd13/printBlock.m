function printBlock(title, len)

if nargin==1
    len=50;
end

n=length(title);
s1=repmat('_',1,floor((len-n)/2));

% if s_text
fprintf('%s%s%s\n',s1,upper(title),s1);
% else
%     fprintf('%s%s%s\n\n',s1,repmat('_',1,length(title)),s1);
end

