%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PURPOSE:      mex all C++-code relevant to the project
%
% AUTHOR:       Per Siden
%               Division of Statistics and Machine Learning
%               Department of Computer and Information Science
%               Linkoping University      
%
% FIRST VER.:   2017-05-22
% REVISED:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Mex them all

close all, clear all, clc

is64 = ~isempty(strfind(computer,'64'));
cd('util/')

% QinvCond QinvCondBreakEarly
if (is64)
    mex -largeArrayDims QinvCond.cpp;
    mex -largeArrayDims QinvCondBreakEarly.cpp;
else
    mex QinvCond.cpp;
    mex QinvCondBreakEarly.cpp;
end

% Qinv
if (is64)
    mex -largeArrayDims Qinv.cpp;
else
    mex Qinv.cpp;
end
