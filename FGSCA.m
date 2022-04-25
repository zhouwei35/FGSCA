
% Spiral Gaussian Mutation Sine Cosine Algorithm for Optimization Problems

function [X_best, Convergence_curve]=FGSCA(N,MaxFEs,lb,ub,dim,fobj)

%　Initialization population
fes = 0;
t=1;
X=initialization(N,dim,ub,lb);
Destination_fitness=zeros(1,dim);
for i=1:size(X,1)
    Destination_fitness(1,i)=fobj(X(i,:));
    fes=fes+1;
end
[Destination_fitness_sorted n]=sort(Destination_fitness);
Flames=X(n,:);
X_best=Flames(1,:);

% Main operation
while fes<=MaxFEs
    if t>1
        %  Evaluate population 
        [Destination_fitness,fes]=get_fitness(X,fobj,fes);
        double_population=[X;Flames];
        double_fitness=[Destination_fitness Destination_fitness_sorted];
        [double_fitness_sorted n]=sort(double_fitness);
        double_sorted_population=double_population(n,:);
        Destination_fitness_sorted=double_fitness_sorted(1:N);

        %  Update the Flames and X_best
        Flames=double_sorted_population(1:N,:);
        X_best=Flames(1,:);
    end
    X_bestscore=Destination_fitness_sorted(1);
    X_best=Flames(1,:);

    %  Sine cosine updating 
    a = 2;
    r1=a-fes*((a)/MaxFEs); 
    for i=1:size(X,1) 
        for j=1:size(X,2) 
            r2=(2*pi)*rand();
            r3=1;
            r4=rand();
            if r4<0.5
                X(i,j)= X(i,j)+(r1*sin(r2)*abs(r3*Flames(i,j)-X(i,j)));
            else
                X(i,j)= X(i,j)+(r1*cos(r2)*abs(r3*Flames(i,j)-X(i,j)));
            end
        end
    end

    %  Spiral motion
    F=spiral_motion(X,Flames,fes,MaxFEs);
    F=Overout(F,lb,ub);

    % Gaussian mutation
    delta = 1;
    G=gauss(X,delta);
    G=Overout(G,lb,ub);

    % evaluation + greedy_selection
    [S,fes]=greedy_selection(F,G,fobj,fes);
    
    X = S;
    Convergence_curve(t)=X_bestscore;
    t=t+1;
end
end

% Get fitness
function [Y,fes]=get_fitness(X,fobj,fes)
y=zeros(1,size(X,1));
for i=1:size(X,1)
    y(1,i)=fobj(X(i,:));
    fes=fes+1;
end
Y=y;
end  

% Spiral motion 
function X=spiral_motion(X,Flames,fes,MaxFEs)
n=size(X,1);
l=round(n-fes*((n-1)/MaxFEs));
c=-1+fes*((-1)/MaxFEs);
for i=1:n
    for j=1:size(X,2)
        if i<=l 
            distance_to_flame=abs(Flames(i,j)-X(i,j));
            b=1;
            k=(c-1)*rand+1;     
            X(i,j)=distance_to_flame*exp(b.*k).*cos(k.*2*pi)+Flames(i,j);
        end
        if i>l
            distance_to_flame=abs(Flames(i,j)-X(i,j));
            b=1;
            k=(c-1)*rand+1;
            X(i,j)=distance_to_flame*exp(b.*k).*cos(k.*2*pi)+Flames(l,j);
        end
    end
end
end  

% Avoid exceeding the upper and lower bounds
function X=Overout(X,lb,ub)
for i=1:size(X,1)
    Flag4ub=X(i,:)>ub;
    Flag4lb=X(i,:)<lb;
    X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
end
end 

% evaluation and greedy_selection 
function [X,fes]=greedy_selection(X1,X2,fobj,fes)
n=size(X1,1);
for i=1:n
    y_X1=fobj(X1(i,:));
    y_X2=fobj(X2(i,:));
    y_col=[y_X1,y_X2];
    [~,k]=min(y_col);
    fes=fes+2;
    switch k
        case 1
            X1(i,:)=X1(i,:);
        case 2
            X1(i,:)=X2(i,:);
    end
end
X=X1;
end  

% Gaussion mutation
function G=gauss(X,delta)
n=size(X,1);
for i=1:n
    G(i,:)=X(i,:)*(1+delta*randn);
end
end

% Initialize 
function Positions=initialization(SearchAgents_no,dim,ub,lb)
Boundary_no= size(ub,2); 
% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end
% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end
end
