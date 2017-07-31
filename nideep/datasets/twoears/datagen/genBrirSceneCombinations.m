function sceneCombinations = genBrirSceneCombinations( n )

for ii = 1 : n
    sceneCombinations(ii).nSources = randi( 4 );
    speakerRandperm = randperm( 4 );
    sceneCombinations(ii).brirSrcIds = speakerRandperm(1:sceneCombinations(ii).nSources);
    sceneCombinations(ii).headPosIdx = randi( 4 );
    sceneCombinations(ii).headBrirAzm = single( randi( 36 ) ) / 36;
    for jj = 2 : sceneCombinations(ii).nSources
        sceneCombinations(ii).snr(jj) = randi( 41 ) - 21;
        sceneCombinations(ii).onlyGeneral(jj) = logical( randi( 2 ) - 1 );
    end
    sceneCombinations(ii).normalizeLevel = 10^( randn(1)*0.7 - 0.35 );
    sceneCombinations(ii).ambientWhtNoise = randi( 4 ) > 2;
    sceneCombinations(ii).whtNoiseSnr = randi( 26 ) - 6;
end

end
