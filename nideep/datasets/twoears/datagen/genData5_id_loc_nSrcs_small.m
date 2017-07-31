function genData5_id_loc_nSrcs( dd, scps, cc, ffs, stopAfterProc )

startIdentificationTraining();

classes = {{'alarm'},{'baby'},{'femaleSpeech'},{'fire'},{'crash'},{'dog'},{'engine'},{'footsteps'},...
           {'knock'},{'phone'},{'piano'},{'maleSpeech'},{'femaleScream','maleScream'}};

featureCreators{1} = 'FeatureCreators.FeatureSet5RawRmAms2Ild';
azmLabeler = LabelCreators.IdAzmDistributionLabeler( 'angularResolution', 5, ...
                'types', classes);
numSrcsLabeler = LabelCreators.NumberOfSourcesLabeler( 'srcMinEnergy', -22 );
datasets = {'learned_models/IdentityKS/trainTestSets/NIGENS160807_75pTrain_TrainSet_1.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS160807_75pTrain_TestSet_1.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS160807_75pTrain_TrainSet_2.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS160807_75pTrain_TestSet_2.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS160807_75pTrain_TrainSet_3.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS160807_75pTrain_TestSet_3.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS160807_75pTrain_TrainSet_4.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS160807_75pTrain_TestSet_4.flist'
            };

brirs = { ...
    'impulse_responses/twoears_kemar_adream/TWOEARS_KEMAR_ADREAM_pos1.sofa'; ...
    'impulse_responses/twoears_kemar_adream/TWOEARS_KEMAR_ADREAM_pos2.sofa'; ...
    'impulse_responses/twoears_kemar_adream/TWOEARS_KEMAR_ADREAM_pos3.sofa'; ...
    'impulse_responses/twoears_kemar_adream/TWOEARS_KEMAR_ADREAM_pos4.sofa'; ...
    };

scc = load( 'brirSceneCombinations260916.mat' );
scc = scc.sceneCombinations;

if nargin < 3, cc = 1 : size( labelCreators, 1 ); end
if nargin < 4, ffs = 1 : numel( featureCreators ); end
if nargin < 5, stopAfterProc = inf; end

doneCfgs = {};
if exist( 'genData5_id_loc_nSrcs.mat', 'file' )
    filesema = setfilesemaphore( 'genData5_id_loc_nSrcs.mat' );
    load( 'genData5_id_loc_nSrcs.mat' );
    removefilesemaphore( filesema );
end

for ff = ffs
for ll = cc
    
    fprintf( '\n\n============== data gen; ll = %d, dd = %d, ff = %d. ==============\n\n', ...
        ll, dd, ff );
    
    pipe = TwoEarsIdTrainPipe( 'cacheSystemDir', [getMFilePath() '/../../../idPipeCache'] );
    pipe.blockCreator = BlockCreators.MeanStandardBlockCreator( 0.5, 1./3 );
    pipe.featureCreator = feval( featureCreators{ff} );
    pipe.labelCreator = LabelCreators.MultiLabeler( {azmLabeler, numSrcsLabeler} );
    pipe.modelCreator = ModelTrainers.LoadModelNoopTrainer( 'noop' );

    pipe.trainset = datasets{dd};
    pipe.setupData();
    
    sc = SceneConfig.SceneConfiguration.empty;
    for scp = scps
%         if any( cellfun( @(x)(all(x==[dd ll ff scp])), doneCfgs ) )
%             continue;
%         end
        
        sc(end+1) = SceneConfig.SceneConfiguration();
        sc(end).addSource( SceneConfig.BRIRsource( ...
            brirs{scc(scp).headPosIdx}, 'speakerId', scc(scp).brirSrcIds(1), ...
            'data', SceneConfig.FileListValGen( 'pipeInput' ) )...
            );
        for jj = 2 : scc(scp).nSources
            if scc(scp).onlyGeneral(jj)
                data = SceneConfig.FileListValGen( ...
                    pipe.pipeline.trainSet('fileLabel',{{'type',{'general'}}},'fileName') );
            else
                data = SceneConfig.FileListValGen( pipe.pipeline.trainSet(:,'fileName') );
            end
            sc(end).addSource( SceneConfig.BRIRsource( ...
                    brirs{scc(scp).headPosIdx}, 'speakerId', scc(scp).brirSrcIds(jj), ...
                    'data', data,...
                    'offset', SceneConfig.ValGen( 'manual', 0.25 ) ), ...
                'snr', SceneConfig.ValGen( 'manual', scc(scp).snr(jj) ),...
                'loop', 'randomSeq' ...
                );
        end
        sc(end).setLengthRef( 'source', 1, 'min', 20 );
        sc(end).setBRIRheadOrientation( scc(scp).headBrirAzm );
        sc(end).setSceneNormalization( true, abs( scc(scp).normalizeLevel ) );
        
        if scc(scp).ambientWhtNoise
            sc(end).addSource( SceneConfig.DiffuseSource( ...
                    'offset', SceneConfig.ValGen( 'manual', 0 ) ),...
                'loop', 'randomSeq',...
                'snr', SceneConfig.ValGen( 'manual', scc(scp).whtNoiseSnr )...
                );
        end

        doneCfgs{end+1} = [dd ll ff scp];
    end
    pipe.init( sc, 'gatherFeaturesProc', false, 'hrir', [], 'fs', 16000, 'stopAfterProc', stopAfterProc );
    
    pipe.pipeline.run( 'modelPath', ['genData5_id_loc_nSrcs' buildCurrentTimeString()], ...
                       'runOption', 'onlyGenCache' );
    
    
    filesema = setfilesemaphore( 'genData5_id_loc_nSrcs.mat' );
    if exist( 'genData5_id_loc_nSrcs.mat', 'file' )
        fileupdate = load( 'genData5_id_loc_nSrcs.mat' );
        for ii = 1 : numel( fileupdate.doneCfgs )
            if ~any( cellfun( @(dc)(all(dc==fileupdate.doneCfgs{ii})), doneCfgs ) )
                doneCfgs{end+1} = fileupdate.doneCfgs{ii};
            end
        end
    end
    removefilesemaphore( filesema );
    save( 'genData5_id_loc_nSrcs.mat', ...
              'doneCfgs' );
      
end
end

end
