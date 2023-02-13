%testmat2pkl.m
% testing a funsdtion to save mat files as pickle files

clear variables

a = 1:10;
b = 1:0.3:3;

%mat2np(b,'test.pickle', 'float64')

load('/Users/wsb/Library/CloudStorage/OneDrive-SharedLibraries-UniversityofSouthampton/Phase retrieval with neural nets - Documents/nanoparticle Mie scattering project/MieScatteringData/20220916/RadialIntegralData3.mat')
% note that Rhys's logs are base 1, bshould convert them to make the scales
% the same
for ii = 1:size(RadialIntegral,1)
    intensities(ii,:) = log(10) * RadialIntegral{ii}.RadSum;
    angles(ii,:) = log(10) * RadialIntegral{ii}.Angles;
end



pickle_it =0
if pickle_it ==1
mat2np(intensities, 'rhys_int2.pickle', 'float64')
mat2np(angles, 'rhys_angles2.pickle', 'float64')
end

writematrix(intensities, 'rhys_intensities.txt')
writematrix(angles, 'rhys_angles.txt')
