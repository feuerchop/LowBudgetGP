function GPNeilPackage(isloaded)

if isloaded
    disp('Add all dependicies of Neil package!')
    addpath ~/Documents/crowdexp/GPN/FGPLVM0p151/
    addpath ~/Documents/crowdexp/GPN/HGPLVM0p1/
    addpath ~/Documents/crowdexp/GPN/NDLUTIL0p157/
    addpath ~/Documents/crowdexp/GPN/KERN0p166/
    addpath ~/Documents/crowdexp/GPN/MLTOOLS0p126/
    addpath ~/Documents/crowdexp/GPN/netlab3_3/
    addpath ~/Documents/crowdexp/GPN/OPTIMI0p132/
    addpath ~/Documents/crowdexp/GPN/PRIOR0p13/
    addpath ~/Documents/crowdexp/GPN/MULTIGP0p11/
    addpath ~/Documents/crowdexp/GPN/DATASETS0p136/
    addpath ~/Documents/crowdexp/GPN/GP0p12/
    addpath ~/Documents/crowdexp/GPN/MOCAP0p132/
else
    disp('Remove all dependicies of Neil package!')
    rmpath ~/Documents/crowdexp/GPN/FGPLVM0p151/
    rmpath ~/Documents/crowdexp/GPN/HGPLVM0p1/
    rmpath ~/Documents/crowdexp/GPN/NDLUTIL0p157/
    rmpath ~/Documents/crowdexp/GPN/KERN0p166/
    rmpath ~/Documents/crowdexp/GPN/MLTOOLS0p126/
    rmpath ~/Documents/crowdexp/GPN/netlab3_3/
    rmpath ~/Documents/crowdexp/GPN/OPTIMI0p132/
    rmpath ~/Documents/crowdexp/GPN/PRIOR0p13/
    rmpath ~/Documents/crowdexp/GPN/MULTIGP0p11/
    rmpath ~/Documents/crowdexp/GPN/DATASETS0p136/
    rmpath ~/Documents/crowdexp/GPN/GP0p12/
    rmpath ~/Documents/crowdexp/GPN/MOCAP0p132/
end
end