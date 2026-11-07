# RHPR agent instructions

## Research status

- Failure repair2026-11-07: original plateau kernel64500ca852e7 ERRORafter
 4modelsfinished. Post-fitguard incorrectly compared successfulAdamsteps to
 scheduledbatches;correct invariant updates+AMPskips=ceil(N/batch)*epochs.
 SUM42/GN42 both60/17760steps/0skip;SUM101100/29593steps/7skips,best80;
 GN10165/19238steps/2skips,best45. All4meetoperationalES. 202neverstarted.
 Independentlyreplayed57VALfiles;allmetrics/counts/cases/SIDs match1e-12.
 Fourlast/bestweightsdownloaded/hashed/metadata verified. Sourcefailedplan/
 config,4checkpoints+sidecars+57predictions83filespinned. No successmanifest
 existsforfailedparent;customrecoveryverifiesallpinnedbytes. FailureZIPsaved.
 FinishONLYoriginal2pending202continuationsfromfull30;reuse4finishedmodels
 withoutexecution;weights/predictionsremaininoriginalfailedkernel/evidenceZIP. Same6matrix,min60/max100,VAL5,patience20,LR.001/FP16/SDPA512.
 No newseeds/tokenizer/proxyfits/TEST/HPO/hardwarefallback/notifications.
 Newprivatecarasik/rhpr-ml1m-proxy-plateau-finish preservesfailedv1parent;
 original3datasets+3successfulkernel sources. Suite5500s/perfit5000/server100min;
 quota41.74h,expected35–85min. Nativeactual65epoch2skiprecoverycopiedallbytes
 withtrainingfunctionpatchedtofailifcalled. Regression1032tests91skips0fail;
 targeted21pass;all4externalbranchesforbidmodelconstruction/training. Configml1m_proxy_plateau_finish_2t4.json;docs
 gr-proxy-plateau-recovery.md/.preflight. FirstfinishCI42376103455 at3ec6719d
 rejectedbySaveKernelHTTP400beforeanysession/GPU(no detailedcause). Revised
 configremovesthefailedfourthinputandembedsverified4selection/curverecords,
 with83originalartifacthashesandexternal-checkpointreferences. Only2pending
 202modelstrain;finalCPUreplayjoins4original+2newpredictions. Preparedfor
 onecorrectedsubmission;no completedtraining repeated.
 TwoCIfinishrequests42376103455/42376580202plusdirectSDKdiagnosticwere rejected
 beforeanyGPU/version. ExactAPIbody:kernel source must be less than1megabytes.
 Confirmedpayload1226614bytes. Globalzlib-compressedSTOREDzip+minifiedJSON
 reducesrunner~941484bytes;all210sourcefiles/configvaluespreserved. Local1MiB
 sizeguardadded;standalonebaselineexecution/fullrecoverybyteauditpass,
 targeted11packaging/recoverytestsandfull1032tests91skips0fail. NoMLchange.
 Fourth-inputhypothesiswasnotestablished;rootconfirmedsource-size limit.
 Thisrepairs/completestheoriginalsix-modeltask;no repetitionof4completedfits.


- User authorization2026-11-07: continue SIX fullTIGERs from epoch30 of private
  carasik/rhpr-ml1m-proxy-target/1 to operational VALplateau. Same seed161803,
  SUM/GradNorm indexes x42/101/202,data/FP16SDPA512/AdamLR.001 unchanged.
  Exact model+Adam+scaler+CPU/CUDA/Python/NumPyRNG resume;12last/bestweights
  SHA/sizepinned,all6metadata30epochs/8880updates/95Adamstates/1CUDA RNG checked.
  Min60/max100totalepochs,VAL5,patience20epochsbeyondbest;NOnewLRscheduler.
  EScriterion=operationalplateau;epoch100withoutcriterion=censored,notconverged.
  Preserve fixed30;newtargetbestVALundercontinuationrule. NoTEST,proxy/tokenizer
  fits,oldseedfits,HPO/retries/extensions/Colab/newuploads/Telegram.
  Existing3datasets+3privateparents,2T4,3resumes/worker. Expected1.5–4h session;
  perfit5000s/suite15500s/server270min;quota43.43hobserved. Nativeactualepoch30
  model/optimizerrestore+tiny30->60ES/RNGprobe pass.1027tests91skips0fail,
  targeted18pass. Configexperiments/ml1m_proxy_plateau_2t4.json;
  docs/gr-proxy-plateau.md/.preflight;privatecarasik/rhpr-ml1m-proxy-plateau.
  SUBMITTEDONCEprivatev1ERROR,executable64500ca852e7,workflow42366590879
  successsubmit-only;210embeddedfiles/exactconfig/T4/private/offline/3datasets/
  3parentsverified. Receiptdocs/gr-proxy-plateau-submission.json. ERROR due
  to incorrect post-fit AMP accounting;four validcompletedmodels preserved,
  two202notstarted. See latest recovery entry;do not repeat completed fits.


- User authorization2026-11-07: cheap independent full-TIGER proxy diagnostic.
  Exactly6fresh generators, existing SUM/GradNorm indexes x tokenizer42/101/202,
  newcommon generatorseed161803,30epochs/VAL5/FP16/nativeSDPA512,2T4.
  Reuse narrow mini epoch10scores from31415/27182;no mini/tokenizer/content fits.
  Original full architecture4+4,d128,FF1024,H6,kv64,4977920params.
  Same151166TRAIN/2786VAL/history20,8880updates/sixVALrequired;NO TEST.
  Primary fixed newfull30;oldfull30 reference stability and earlynewfull5/10
  comparators;earlyfull shares targettrajectory,not independent transfer.
  Structured6candidate subset fixed bylabels,alreadyexamined MTLindexes andVAL;
  not a newtokenizerfamily benchmark orconvergenceclaim. Reportshortlist/regret,
  seedgroups/methodgroups,pins andallvalidationpredictions. Existing3datasets/
  2privateparents only,no dataupload/HPO/extensions/retries/Colab/Telegram.
  Native6initializersagree4cfe16ae1433c48e8d7aad7531c82af7da3d370f7f5baf25b8d7259d12d39dc6;
  2tiny native30epoch/6VALprobespass;1025tests91optional skips0fail,
  finaltargeted16checks pass. Private carasik/rhpr-ml1m-proxy-target.
  Config experiments/ml1m_proxy_target_replication_2t4.json;
  docs/gr-proxy-target-replication.md andgr-proxy-target-preflight.json.
  Observedquota45.00h GPU;expected1.5–2h session;server135min/suite7500s/
  perfit2300s. SUBMITTEDONCEprivatev1RUNNING,executable5ed498804545,
  workflow42354986331successsubmit-only;209embeddedfiles/exactconfig/
  3datasets/2parents/T4/private/offlineverified;receipt
  docs/gr-proxy-target-submission.json. COMPLETE6/6fits,93.343min,all30epochs/
  8880updates/6VAL/0AMPskips. Same6indices mini-oldfullrho.8857 becomes
  mini-newfull-.0286;old/newfull.2571;earlynewfull10/new30.0286. Mini selects
  SUM202(.065194),newwinnerGN202(.070583),regret.005390/7.64%;paireduserCI
  [-.001107,.012239]cross0(fixedmodels,posthocpair,no seed/selectioncorrection).
  Mini top3retainswinner,notstrongwith6candidates. Fournewbestepochs30;
  no convergenceclaim. Posthoclate-meanrho.486,twofullseedmean.657;not new
  confirmation. SameVAL/relatedindices;oldfullseeds coupledtoindexseedgroups,
  newcommon161803. No validatedportableproxy or isolatedarchitecturefailure.
  Independent54predictionreplays(36new+12mini+6oldfull),allmetrics/pins/budgets
  exact. Meanfullfit30.44min,epoch58.61s. Reportsgr-proxy-target-results.md/
  .json,replayanalyze_proxy_target.py,plots/numericalevidenceZIPsaved.
  No GPUresults pending. No additionalruns/extensions/HPO/notifications.


- User authorization2026-11-06: projectioncontrol plus additionalfullytrainable
  RQ-VAE NOprojection arm. Userclarified ordinaryRec+VQloss shouldadaptencoder/
  decoder to sharedbooks;not a directNIPgradientthroughargmin. Twofreshfits,
  cleanML1M151166TRAIN,seed42,30epochs/VAL5,FP16SDPA512,2T4,~40minbudget.
  projected_frozen:original64DRQfullyfrozen,fixedSIDs,trainsharedLinear64->128.
  joint_rq:NOprojection,expandnativeencoderlast/decoderfirst/books64->128
  withzeros,functionpreservinginitially;actualcentroidParametersserveTIGER
  lookup;allRQtrainable. NativeSTE/beta.25meanloss;objectiveNIP+Rec+VQ1/1/1,
  uniformcatalogue512auxsample/update,FP32RQmath;Adam.001/wd.0001sameasTIGER.
  SIDsreassignedeveryepochstartfromcurrentencoder/books,epochSIDbufferrestores
  bestmatchingindexbeforeTEST. Captureobjectivecomponents/churn/collisions,
  failedcandidatecodes/books/latents.NoHPO/retries/extensions/Telegram/newdata.
  SAMEinitialTIGERlookup/baseTransformer,pinsbyarm;nominal/activeparams
  projected8157123/4887808,joint8214531/8214531. ExistingprivatecleanMTLparent
  and3datasetsreused;no content/RQpretraining. Server37min,fit2050s,quota.65h.
  Private carasik/rhpr-ml1m-codebook-adaptation;config
  ml1m_codebook_adaptation_2t4.json,docs/gr-codebook-adaptation.md.
  2native30epoch/6VALprobespass;encoder/decoder/booksauxgradientsverified;
  1021tests91optional skips0fail. Noadditionalfitsauthorized.
  SUBMITTEDprivatev1RUNNING,executableb828588a3bb1,workflow42217880480success.
  All208embeddedfiles/config/3datasets/1parentexact;receipt
  docs/gr-codebook-adaptation-submission.json. COMPLETE2/2fits,33.951min,
  both30epochs/8880updates/6VAL/0AMPskips. ProjectionVAL.03686649/TEST.03114151
  selected30;fulljointVAL.02118170/TEST.01895965selected25. Historicalcontrol
  VAL.06665966/TEST.06165244;one seed,no robustgain. Jointindexepoch30occupied
  27/24/30,1756uniquesemanticIDs,maxcollision9,medianSIDchurn84.93%;RecLoss
  2.6305->2.3620,VQ.1850->1.4835. RQsupportavoidsoldoverflow,rankingstillpoor.
  Independent14VAL/TESTpredictionfilesrescored,casehashes/pins/budgetsmatch.
  TRAINtargetentropyoriginalincrementbits7.6601/2.7691/.0579/.00072 vsjoint30
  4.4168/3.2097/2.2208/.6405,total10.48779both(fullSIDbijection);thisisnot
  user-historyconditionalNIPLoss.No measuredgradientconflictclaim.
  Report gr-codebook-adaptation-results.md/.json,raw/CPUreplay/plots/33MBevidence
  ZIPsaved. Docs-onlycollection,no newsourceorGPUwork;1021predeploymenttests
 91optional skips0fail. No GPUresults pending;no furthercompute authorized.


- User authorization2026-11-06: two shared-codebook TIGERfits on existingclean
  ML1M151166TRAINwindows,seed42,30epochs,val5,FP16/nativeSDPA/512,2T4.
  Reuse SUMseed42 frozenRQ-VAE andcontent fromprivatecleanMTLparent.
  Arm codebook_init copies3x256centroids into TIGERinputtable,keepsSIDfixed;
  arm live_codebook uses SAMEactualtrainable768rows as RQcentroids and
  reassignsSID atSTARTeveryepoch (userexplicitlyselectedthis). FrozenRQencoder
  latents cachedonce;decoderunused/frozen,noRecLoss,noargmingradient.
  64Dbooks/latentszero-liftto128D preservesinitialassignments;all128Drows
  thenlearn.NostandardGRIDoutputheadtyingadded.Suffix/user/BOSnormal.
  PersistentactiveSIDbuffer ensuresbestcheckpointrestoresmatchingindex.
  Exact30epochs/8880updates/6VAL,onebestVALselectedTESTeach;saveeveryepochSID,
  churn/collisions. Initialmodelsidenticalpindf5affc3c6ae40678b80a58deb62d162c51019187790842af5f2e7bc7ac83e6a.
  ExistingrandominitSUMcontrolreused;NOthirdfit or newquantizertraining.
  Private carasik/rhpr-ml1m-shared-codebooks,offline,3datasets/1parent;
  no newdataupload,retries/HPO/Telegram/hardwarefallback.Quota1.18hchecked;
  server60min/fitbudget3000s. Config ml1m_shared_codebooks_2t4.json,
  docs/gr-shared-codebooks.md. Native2x30epoch/6VALprobesandfrozenencoder
  sourcecheckpass;selectedearlierepoch20/25SIDrestoreverified.
  SUBMITTEDprivatev1RUNNING,executablef6a299f5bb60,workflow42210018539success.
  All206embeddedfiles/config/3datasets/1parentexact;1015tests91skips0fail.
  Receipt docs/gr-shared-codebooks-submission.json. COLLECTEDERROR:staticarm
  completes30epochs/8880updates/6VAL/0AMPskips,VAL.059715/TEST.051880(-15.85%
  vshistoricalSUMcontrolTEST.061652). Live29epochs/8584updates/5VAL/0AMPskips,
  failed atstart30:264itemcollision exceeds256suffix. No liveTEST/topmarkers.
  MedianSIDchurn95.81%;failedindex244uniquesemanticIDs/2874items,used21/46/64.
  Posthocdropadded64coordsgives270uniqueIDs/maxgroup254;collapsepersists,
  geometryliftcontributesbutisnotsolecause.NoRecLosswastrained;NIP-only
  embeddingupdatesdonotpreservefrozenencodergeometry.FirstbookRMS7.32->.6265.
  Independentlyreplayed19predfiles,epoch29checkpointandfailedindexrecomputed.
  1017tests91optional skips0fail. Report gr-shared-codebooks-results.md/.json;
  replay/plots/labeledlogs/62.4MBevidenceZIPsaved.Loggingfixedwithtrialevents
  andfailedcandidate snapshotsforfutureonly;NOnewGPUrun/retryauthorized.
  No GPUresults pending;do not submitduplicates.


- User authorization2026-11-06: two smaller mini-TIGER shapes ffn_small
  (4+4,d64,FF256,H6,kv32,1112896params) and attention_small
  (4+4,d64,FF256,H4,kv16,719552params).48freshfits:12frozenSUM/UW/GN/PC
  indexes seeds42/101/202 x2shapes xgenseeds31415/27182.10epochs,val5,
  fixedepoch10primary vs savedfull30VAL;reportoverall,seedgroups,eachmethod.
  Reuse savednarrow controls;no retraining baseline. Private carasik/rhpr-ml1m-mini-width,
  2T4,offline,3existingdatasets,2existingparentkernels;nonewdataupload.
  No TEST,quantizer/fullmodel/content fitting,Telegram,retries/HPO/extensions.
  Internal9900s/server180min,quota3.52hchecked. Config
  ml1m_mini_tiger_width_2t4.json;protocol docs/gr-mini-tiger-width.md.
  Supersedes prior no-extra-shapes only for these48fits. Do not repeat oldruns.
  SUBMITTED privatev1 RUNNING,executabled80787c82e1c,workflow42182356635success.
  All205embeddedfiles/config/3datasets/2parents verified.1012tests91skips0fail;
  4native10epoch/2VALprobes pass. Receipt docs/gr-mini-tiger-width-submission.json.
  COMPLETE48/48fits,139.756min,10epochs/2960updates/2VAL/0AMPskips each.
  Independently replayed96new+48narrow+12fullVALfiles;all metrics/pins match.
  Narrowall12rho.902,SUMrho1;ffnsmallallrho.385,SUM.5;attention_smallall-.126,
  SUM1butindividualseed.5/1. Epochseconds38.85/36.37/30.19;only1.07/1.29x
  speedvsnarrow. Narrowwithinindexseed101rho-.2;poolednotuniversalproxy.
  Retainnarrowprovisionally;no robust generalreplacement fromsmallerwidths.
  1012tests91optional skips0fail;report gr-mini-tiger-width-results.md/.json,
  replay scripts/research/analyze_mini_width.py,plotsand57.2MBevidenceZIPsaved.
  No GPUresults pending,newcompute,extensions or retries authorized.


- User authorization2026-11-06: fixed narrow mini-TIGER epoch10 transfer on
  FOUR saved SUM/UW/GN/PC indexes at tokenizerseed202.8freshfits crossed
  with genseeds31415/27182,2T4;exact10epochs/val5,primary fixedepoch10.
  Same narrow4+4d64/d_ff512/6heads/d_kv32,initialization pins and all other
  data/training settings as pilot. No other shapes/seeds/HPO/epoch extension.
  Reuse only existing private cleanMTL-replicates parent,3existingdatasets;
  NO new data upload,quantizer/content/fullmodel training,TEST or Telegram.
  Config ml1m_mini_tiger_transfer_2t4.json,phase seed202_transfer;private
  carasik/rhpr-ml1m-mini-transfer.60minservercap,3000s internaldeadline.
  Submitted private carasik/rhpr-ml1m-mini-transfer/1 COMPLETE,executable31d120c8,
  workflow42174475061success;all205embeddedfiles/config/3datasets/1parent
  exact.1009tests91optional skips0fail;2native10epoch/2VAL probes pass.
  COMPLETE8/8fits,10epochs/2960updates/2VAL/0AMPskips,27.823min,finished
  2026-11-06T06:55:41.629635UTC.Fixedepoch10meanrho.8,top1/top2correct,
  targetregret0;genseeds1.0/-.2,singlebadseed selects worstUW.Conditional
  userCI[-.6,1.0],fourindexes/exactpermutationp.333;no universalproxy claim.
  Independently rescored16mini+4fullVALfiles;allpins/cases/metrics match.
  Epochseconds38.827,meanfit403.96s;report gr-mini-tiger-transfer-results.md
  /.json,replay/plots/evidenceZIPsaved.No GPUresults pending or newcompute.
  Receipt docs/gr-mini-tiger-transfer-submission.json;do not launch duplicates.
  Protocol docs/gr-mini-tiger-transfer.md. Prior no-followup restriction is
  superseded only for these8fits;do not repeat or resume the failed pilot.


- User authorization2026-11-05: frozen-index mini-TIGER calibration. Eight
  existing SUM/UW/GN/PC indexes at tokenizerseeds42/101;three mini shapes
  shallow1+1,narrow4+4d64,compact2+2;generatorseeds31415/27182 only.48fresh
  fits,20epochs,val5,FP16/nativeSDPA/512,2T4;no quantizer/fullmodel fitting.
  Reuse existing private cleanMTL and replicates notebook outputs as kernel
  sources;all reference files/manifests/pins verified. NO new dataset upload:
  that attempted action was rejected by automatic approval review and abandoned.
  Submitted private carasik/rhpr-ml1m-mini-tiger/1 ERROR,executable44b84d5d,
  workflow42109946957success;all205embeddedfiles/config/3datasets/2parents
  exact.1005tests91optional skips0fail;3native20epoch/4VAL probes pass.
  Collected ERROR after171.23min:36/48fits complete,all20epochs/4VAL/0AMPskips.
  Shallow16+narrow16 cover8indexes/2genseeds;compact4coversonly2indexes.
  Two internaldeadline TimeoutErrors,10neverstarted;no retry/newcompute.
  All144miniVALfiles+8full30VALindependently rescored. Narrow10 pooledrho.833
  (genseeds.857/.810),but indexseedgroups.8/-.2,conditionaluserCI[-.024,.905].
  Shallow20pooledrho-.167;do not infer depthcausality or universalproxy.
  Meanepochseconds shallow16.96,narrow38.82,fullhistorical57.91.
  Earlyfull5pooledrho.857shares targettrajectory;not independenttransfer.
  Reports gr-mini-tiger-results.md/.json;replay,plots,evidenceZIPsaved.
  Wholequality incomplete,analysis complete,noGPUresults pending;1005tests,
  91optional skips,0fail.No follow-up authorized.
  Receipt docs/gr-mini-tiger-submission.json;do not launch duplicates.
  Protocol docs/gr-mini-tiger.md;config ml1m_mini_tiger_2t4.json. Primary
  correlation against full epoch30VAL;best-of30 secondary;report seed groups
  and earlyfull comparators.8related indexes,one fullgen seed/index;pilot only.
  No TEST evaluation,extra shapes/seeds/epochs/HPO/retries/Telegram authorized.
  Server3h budget;incomplete matrix must fail,never report as full calibration.


- Latest authorization2026-11-05: ALS GradNorm + ALSinput lambdaCF0,
  seed42,2freshfits on2T4;config ml1m_als_objectives_2t4.json,
  private carasik/rhpr-ml1m-als-objectives. Same30epochs/val5/FP16SDPA512,
  AE3500/RQ3000FP32,all5pinnedinputs reused. ALS remains active input in
  botharms;zeroCF uses fixedweights1/1/1/0 with NO MTLadapter,otherwise
  adapter would override fixedweights. GradNorm4tasks alpha1.5,sumweights4.
  Tinygradientcheck verifies liveALSencoder vs zeroCFdecoder reconstruction
  gradient;bothnativeE2E30epoch/6VAL passed. Reuse old contentSUM/ALSSUM controls.
  No extra methods/seeds/HPO/retries/Colab/hardwarefallback/Telegram.
  COMPLETE privatev1,executablea102ceac9a37,workflow42091502981.
  All204embeddedfiles/config/5sources/T4/offline verified.1000regressiontests,
  91optional skips,0fail;2nativeE2E+explicitzeroCFgradientcheck passed.
  Bothfits30epochs/8880updates/6VAL/0AMPskips,37.53min. GNselected25,
  lambda0selected20. VAL/TEST GN .062935/.053425,lambda0 .065669/.048492;
  TESTvsALSSUM -3.07/-12.02%,vscontent -13.34/-21.35%. No positive evidence.
  GNfinalweights meta/poster/plot/ALS .878/.930/1.127/1.065;CF not suppressed.
  Actual lambda0 quantizer audited:CFdecoder unchanged,CFencoder trained;
  masking CF changes all2777supportedlatents,medianrelativeL2 7.25%.
  35predictionfiles independently rescored. One seed;vsALSSUM CIs include0;
  vscontent conditional userCIs negative,not trainingseed inference.
  Receipt docs/gr-ml1m-als-objectives-submission.json;no GPUwork pending.
  Report gr-ml1m-als-objectives-results.md/.json,replay/plots/audit/ZIPsaved.
  Docs gr-ml1m-als-objectives.md;do not launch duplicate sessions.


- Latest authorization2026-11-05: TRAIN-only ALSchannel and shuffledALS on
  seed42,ordinarysum,2fresh30epochfits/val5 on2T4;reuse oldSUMseed42control.
  Config experiments/ml1m_als_channel_2t4.json;private carasik/rhpr-ml1m-als-channel.
  SameML1Mrawclean cases/nativeSDPA/FP16/batch512/512,TIGER4+4d128,AE3500/RQ3000.
  Frozen128DimplicitALS on303199uniqueTRAINedges,2777supporteditems;noholdout
  reads. Shuffledonlysupported rows,mask/vector multiset retained. ALSuses
  existingpeople128slot;NOpeopledata. Oldparameter shapes/pins preserved.
  One extra unitRecLoss changes totalscale3to4 in BOTHarms;document limitation.
  Inputprivate carasik/rhpr-ml1m-als-input,manifestcfed1590e72447de047987e45d5399718404a0074eaa6be3bde7ea9086ba9663.
  COMPLETE privatev1,executable6ecbdaa0b54b,workflow42076145430success;
  all204embeddedfiles/config/5sources/T4/offline verified.998regressiontests,
  91optional skips,0fail;2nativeE2Epassed. Submission receipt preserved.
  Bothfits30epochs/8880updates/6VAL/0AMPskips,38.07min;all selectedepoch25.
  VAL/TEST content .066660/.061652,trueALS .064178/.055118(-10.60%TEST),
  shuffled .068415/.059330(-3.77%). TrueALSvs shuffled -7.10%TEST.
  One seed;allconditional bootstrapCIs include0. No gross codecollapse;
  trueALSRecMSE .900 vs shuffled .972,betterRec didnot imply betterranking.
 42predictionfiles rescored incl old4refs;contentfeaturesexact,samepinsverified.
  Report gr-ml1m-als-channel-results.md/.json,replay/figures/evidenceZIPsaved.
  No new compute pending;do not launch duplicates.
  No extra methods/HPO/seeds/retries/Colab/hardwarefallback/Telegram.


- Latest authorization2026-11-05: repeat ALL4 ML1M clean MTL methods on
  two additional pipeline seeds101/202 (not only SUM/PCGrad). Eightfreshfits
  on2T4 in one private carasik/rhpr-ml1m-clean-mtl-replicates session;
  experiments/ml1m_clean_mtl_replicates_2t4.json. Same exact originalrawclean
  2786users/2874films/151166cases,shared3modalities,nopeople,AE3500/RQ3000FP32,
  TIGER4+4d128/FP16/nativeSDPA/batch512/512,30epochs,val5,oneVALselectedTEST.
  All initialization pins precomputed;trial seed reaches quantizer,generator,
  and train permutation via trial-local config copy. Preserve seed42 output.
  COMPLETE v1,executablefc2a62bdd57a,workflow42033993480success.
  All203embeddedfiles/config/hardware/source sets verified;receipt preserved.
  Fullregression995tests:904pass91optional skips,0fail;8native5.22.1E2Epassed.
  All8newfits30epochs/8880updates/6VAL/0AMPskips;146.42min total.
  Combined3seeds42/101/202:meanVAL/TEST SUM .066740/.059173,
  UW .066528/.058160(-1.71%TEST),GradNorm .066932/.060195(+1.73%),
  PCGrad .067324/.058395(-1.32%). PCGradTEST +5.67/-6.28/-4.01% byseed;
  GradNorm -2.44/+12.38/-2.94%. No robust MTLwinner;allconditional userCIs
  cross0. Seed42 apparentPCGradgain didnot replicate. Frozenfeatures exactly
  equal;84predictionfiles independently rescored. Fullreport/replay/ZIPsaved.
  Report gr-ml1m-clean-mtl-replicates-results.md/.json. No new compute pending;
  do not resubmit or add compute automatically.
  Prior no-extra-seeds restriction is superseded only for these8fits.
  Full preflight and run receipt:docs/gr-ml1m-clean-mtl-replicates*. No extra
  HPO,methods,retries,Colab deployment,hardware fallback or Telegram authorized.


- User authorized2026-11-05: fresh ML1Mclean contiguous20+val+test MTLscreen,
  SUM/UW/GradNorm/PCGrad seed42,30epochs,val5,FP16TIGER/nativeSDPA,512/512.
  Config experiments/ml1m_clean_mtl_2t4.json;private carasik/rhpr-ml1m-clean-mtl.
  2786users,2874films,151166TRAINcases;allrawbarriers/singletons checked before
  filtering;casearrays match prioraudit and replay byte-for-byte. Shared once:
  metadataSentenceT5+keyedposter+description-onlyMPNet384chunks,no people.
  AE3500/RQ3000 remainFP32;inactivepeople slot zero preserves oldarchitecture
  and qpin54253fbb;fresh5.22.1generatorpind57fc2e7. TwoT4/fourfreshfits.
  982regressionOK13skip+9newtests+4nativeE2E+2multibatch probes pass.
  COMPLETE private v1,executable20a4c085,workflow42020589648success;
  all203embeddedfiles/config/dataset sources/hardware verified. All4 fits30epochs,
  8880updates/6VAL/0AMPskips;73.34min whole suite,~57s training epoch.
  VAL/TEST NDCG10 SUM .066660/.061652,UW .068528/.056512,
  GradNorm .065226/.060147,PCGrad .070168/.065146 (+5.26%/+5.67% vsSUM).
  One pipeline seed42;all paired-user bootstrap95%CIs cross0;not confirmed.
  PCGrad652/6500projectedupdates (10.03%),489AE+163RQ. UWweights sum3.48.
  28rawpredictionfiles independently rescored;shared cases/features/pins checked.
  Report gr-ml1m-clean-mtl-results.md/.json,CPUreplay/figures/evidenceZIP saved;
  docs/gr-ml1m-clean-mtl-submission.json records version/source/completion.
  No extra seeds/tuning/retries/hardware fallback/Telegram authorized.
  Docs gr-ml1m-clean-mtl.md and preflight JSON;do not launch duplicates.


- ML1M clean-history CPU audit2026-11-05 COMPLETE;NO GPU/protocol change.
  Fixed2874films. Priorfilterproposal5672users/839388cases reproduces.
  799users(14.1%)haveval/test raw timestamp ambiguity hidden by filtering;
  4440(78.3%)have uncovered events in original up-to20val/test contexts.
  Not evidence this caused ML100K quality instability. New raw singleton +
  clean raw up-to20windows:4146users/174378traincases;full20eval:3158/166912;
  literalcontiguous20+val+test:2786/151166. Allcohorts>=1traincasebeforeval.
  Initialvalidprefixonly2600/35730(full20:753/21989). Keep raw timeline/barriers,
  filter examples rather than splice histories;natural short original prefixes
  allowed,never reset-shortafterunknown. Candidate only,not approved training.
  Report gr-ml1m-clean-history-audit.md/.json,CLIreplay/CSV/bundle preserved.


- Batch T4 pilot COMPLETE,collected2026-11-05;carasik/rhpr-t5-batch-pilot v1,
  executable b777b516,570.12s experiment.16attempts:15complete0AMPskips,
  1OOM(eager window50physical1024),0numericalfailed. SDPA20throughput at
  effective/physical256/128,256/256,512,1024:1938/2452/2651/2771examples/s;
  oldeagerbaseline256/1281969. SDPA50:859/910/934/938 vs oldbaseline708.
  SDPA256/256 +24.5%throughput20,+28.4%50;effectivebatch unchanged.
  SDPA512gets95.7%/99.6%of1024throughput with~half memory;larger effective
  batches change updates/epoch and quality is UNTESTED. No defaults changed.
  Replayed750timings+150E2E,independent80268case reconstruction/all6samplehashes.
  Report gr-t5-batch-results.md/.json,replay/figures/evidenceZIP saved.
  No extra GPU/fulltraining/ML1M/Telegram authorized or pending.


- User authorized2026-11-05: bounded batch-throughput pilot, NOT full training.
  experiments/t5_batch_t4_pilot.json; private carasik/rhpr-t5-batch-pilot.
  Old4.47eager vs5.22.1SDPA,windows20/50,effective/physical256/128,256/256,
  512/512,1024/1024;16fresh-process attempts on same physical T4.
  10warmup+50timed+10E2E updates,FP16/dropout.1,LR.001 unchanged.
  Fixed1024TRAINsample prefixes,byte-pinned existing sum42weights/SIDs.
  OOM/numerical failure recorded,never reduce/retry;childbudget1200s.
  User accepts changed effective batches;no quality/NDCG conclusion.
  Submitted v1,executable b777b516,workflow41998201870success;now COMPLETE;
  all202embeddedfiles/config verified,26tests+2nativeCPUbackends pass.
  See docs/gr-t5-batch-pilot.md and gr-t5-batch-submission.json;no duplicates.
  No extras/fullML1M/Telegram authorized.


- Native SDPA pilot COMPLETE,collected2026-11-05: carasik/rhpr-t5-sdpa-pilot v1,
  executable d6edbf78,T4/PyTorch2.11;194.2s experiment,3runtimepaths. Real native
  efficient CUDA forward/backward confirmed. Historical4.47eager→5.22.1SDPA:
  window20 124.67→128.88ms(+3.38%),allocated .841→.747GiB(-11.25%);
  window50 348.85→308.27ms(-11.63%),2.353→1.773GiB(-24.64%). E2E signs same.
  Modern eager itself slower; comparing only modern eager overstates benefit.
  FP32 all95gradients/hidden/beam parity pass;6FP16benchmarks0AMPskips. One
  sequential timing run,not full epochs/convergence/NDCG. Defaults unchanged.
  Reports gr-t5-sdpa-results.md/.json, replay and evidence ZIP preserved.
  No extra runs/full ML1M/quality training/Telegram authorized or pending.


- User authorized2026-11-05: bounded native SDPA T4 pilot, NOT full training.
  Config experiments/t5_sdpa_t4_pilot.json; old4.47eager vs5.22.1eager vsSDPA,
  same trained sum42 checkpoint, windows20/50,10warmup+50timedsteps,FP16/dropout.1.
  CPU parity ALL95gradients/beam passed; no CUDA performance claim yet.
  Modern wheels isolated; private weights mounted from existing Kaggle notebook.
  Submitted private carasik/rhpr-t5-sdpa-pilot v1 RUNNING; executable d6edbf78,
  workflow41993141529success, all201embeddedfiles/config exact.975testsOK13skip;
  13modernGRIDtests/CPUparity pass. Collected COMPLETE; see results entry above.
  See docs/gr-t5-sdpa-pilot.md and gr-t5-sdpa-submission.json.
  No extra runs/seeds/ML1M/Telegram/retries authorized.


- Singleton MTL12fit run COMPLETE,collected2026-11-05MSK,private carasik/rhpr-
  ml100k-singleton-mtl v1,35ce22dd1f07aaf02251af7847eb242f91187413,195.26min2T4.
  All12freshfits30epochs/9420updates/6valchecks/0AMPskips.903singletonusers,
  alloriginalsourcebytehashes/splitselection and3casedigests independently checked.
  Mean VALsum/UW/GN/PC .070893/.065722/.068257/.064754;
  TEST .046652/.049578/.046040/.047780,relativeUW+6.3/GN-1.3/PC+2.4%.
  UWTESTtie42,wins101/202;VALworse2/3(-7.3%mean). PCVALworse3/3. Sum remains
  mean-VALsupported;allconditionaluserCIs cross0,3coupledseeds,sameheldouts.
  Removing tied-targetchoice does NOT remove val/test disagreement. Old/newmetrics
  notdirectlycomparable. PCGrad44/19500calls project(.226%),129pairprojections,
  119inthefirst500AEcalls. All468sampledencoderpaircos positive(min.306).
  GradNormplotweight~1.09-1.13;UWmostlyglobalRecscale,notpurebalanceablation.
  Independentlyrescored84predictionfiles, maps/bookcounts/initpins/transfer/
  checkpointsverified. GN101TEST2invalidpaths,others0,all0duplicates.
  Reports docs/gr-movielens100k-singleton-mtl-results.md/.json,CPUreplay/ZIP
  preserved. No new GPU/model calls/training/inference/dispatch/Telegram pending.


- User authorized2026-11-04: fresh12-model MTL comparison, singleton endpoints,
  sum/UW/GradNorm/PCGrad x42/101/202,window20,30epochs,val5. New private kernel
  carasik/rhpr-ml100k-singleton-mtl;config movielens100k_singleton_mtl_3seeds_2t4.
  Original manifest-verified943-user sequences+embargo derive903users,82234train
  events/80268cases;latest2singleton groups and>=2earlier train timestamp groups.
  3173between-val/testevents CONTEXTONLY,3098aftertestdiscarded;all1273films kept.
  Testhistory=train+val+between,thenlast20. No trainfuture/same-time leakage.
  FreshAE3500/RQ3000/TIGER4+4,d128FP16/FP32beam20test;allseedsfullinitpins.
  2T4;quota15.35h,estimate3–3.5h,watchdog300min. Noextras/retries/Telegram/Colab.
  Per500MTLJSONexport fixed;PCGradcounts aggregateallsteps;optimization unchanged.
  See docs/gr-movielens100k-singleton-mtl.md and preflight JSON. Submissionreceipt
  records actualrunstatus;no qualityresults until existing kernel completes.


- MTL timestamp-target CPU follow-up2026-11-04: original tie-group selection
  remains417val/409testof943. Fixed-checkpoint uniform one-target expectedtest
  sum.042557/UW.036979/GradNorm.049768/PCGrad.040819;relativeUW-13.1%,GN+16.9%,
  PC-4.1%vs sum. GNsingletontest534+15.3%. UniformVAL UW+11.6%,GN-1.7%,PC+1.4%,
  so disagreementpersists. GNpaireduserCI[-.00186,+.01607]includes0,onepipeline
  seed;labeldrawpositive99.56%conditionalNOTpvalue. OriginalVALselectedcheckpoints
  and testhistories unchanged;diagnosticNOTnewbenchmark. NoGPUcalls/training.
  Updated MTLreport, docs/gr-movielens100k-mtl-timestamp-targets.json, replay andZIP.


- MTL seed42 screen COMPLETE2026-11-04MSK, private carasik/rhpr-ml100k-mtl v1,
  executable5af82b106c3d2687459dcb9ebc4cc2e4e1547415,63.60min2T4,4successful
  freshfits,30epochs/10020updates/6valchecks/0AMPskips. No new GPU work pending.
  Sum/UW/GradNorm/PCGrad val .059977/.064748/.057619/.060008;
  test .043635/.039266/.044435/.040150; relative -10.0/+1.8/-8.0%vs sum.
  UWwinsvalbutlosestest;allconditionalpaireduserCIs cross0;only1pipeline seed.
  All13probes/method have positive last-encoder paircos;sum min.708 median.933.
  PCGrad13sampledstepsprojectioncount0;notaggregateallsteps. UWweightssum3.553
  vs3,relative weightsnearuniform;GradNorm .952/.962/1.086. No establishedgain.
  Reconstructed/rescored28predictionfiles, alltemporal casehashes and SIDmaps,
  bookcounts,initialpins,AE/RQtransfer/checkpointsverified. UW6invalidtestpaths,
  others0;all0duplicates. MTLdedicatedJSONnotexported;52eventsrecoveredfromstdout.
  Reports docs/gr-movielens100k-mtl-results.md/.json;CPUreplay/script/ZIPpreserved.
  No extra seeds/tuning/training/inference/Telegram authorized. Colab deferred.


- User authorized2026-11-04: first MovieLens100K MTL screen, metadata+poster+plot,
  NO people, window20, fixed30epochs,val5. Config movielens100k_mtl_2t4.json.
  Seed42 chosen for first screen; four fresh pipelines sum/UW/GradNorm/PCGrad.
  AE3500/RQ3000 fixed, unchanged3x256 and TIGER4+4/d128; frozen prior features,
  native initialization pins, same temporal targets/FP16train/FP32beam20test.
  Docs gr-movielens100k-mtl.md specifies algorithm conventions and UW scale caveat.
  Quantization gradients untouched; save MTL weights/gradient cosines and books.
  No new seeds/graphs/windows/HPO/fallback/retries/Telegram. GPU quota1.43h;
  estimated55-75min2T4. Preserve existing source results. Publication/dispatch
  receipt in docs/gr-movielens100k-mtl-submission.json records actual status.


- Context50 split-gap CPU audit COMPLETE2026-11-04MSK; no GPU/model calls.
  Report docs/gr-movielens100k-context50-split-gap.md/.json. Main:new target-choice
  sensitivity using all timestamp-group films from2042embargoed events. Expected
  one-target graph/control50 val+1.85%,test-4.80% vs official+6.07/-7.99%; fixed-
  model val/test gap shrinks54.8%, not causal attribution or benchmark replacement.
  Context40test+9.77% becomes+1.54%. Same original checkpoints/test history kept.
  Pop x actor-overlap composition explains3.7%of signed gap descriptively;96.3%
  remains. Partial actor-feature correlations<.04. No robust cohort router found.
  Same293Comedy test targets:graph40+27.1%all3seeds,50-30.0%all3.103ofthose users
  have identical unpadded movie lists across widths;fitting/padding still confounded.
  Target choice/checkpoint/user/movie effects matter;no static harmful-genre rule.
  Cross-selected validation halves diagnostic still graph+7.19%;not independent
  because fullval LR scheduling. Per-user deltaVal/Test r=.021;single-item sparse.
  Nice>1h test84users lead only4improved/1worse/79ties,not stableat40/val.
  72cohort interaction simultaneous bootstrap bands allcross0. User/film tables,
  90context50prediction rechecks, allofficial40/50label scores and portable replay
  preserved. Rating timestamps are not viewing. Official artifacts unchanged.
  No training/inference/dispatch/tuning pending;do not spend GPU without new request.


- Context50 replication COMPLETE, collected2026-11-04MSK. Private carasik/rhpr-
  ml100k-context50-replication v1,d8027d0d84d375659335680f63468423718c4a37,
  4fits101/202 in127.75min;reuse completed42,not retrained. All6L50 models30ep/
  10020updates/14valchecks/0AMPskips. Newbest epochs:101control25/graph22,
  202control29/graph28. L50val meancontrol.063543/graph.067401(+6.07%);
  test.046596/.042873(-7.99%). Graph/control test by42/101/202:-10.8/+15.2/-24.8%.
  Val/test graph-control sign reverses each seed. Graph wins val2/3,test1/3.
  Graph20->50val positive3/3,+8.11%mean;controlnegative3/3,-7.65%. Graph40->50
  val+2.18%,test-16.80%;new101/202only test-9.62%. No established overall gain.
  Conditional graph-control50 bootstrapCIs cross0;graph50-40testnegative forall3,
  crosses0fornew2only. Not seed variance, repeated-holdout or multiplicity correction.
  Test knownslotscontrol40.42%/graph40.35%;repeats persist. Control101 has63invalid
  beampaths(20top10),0duplicates;othernewtestsvalid. CPUvalid-path compaction does
  not change itsNDCG10;official ranks unchanged. Epochs123-125s,newpeak2.400-2.402GiB.
  119predictionfiles rescored1e-12;configs/index/init/cases/suffixes/resume audited.
  Report docs/gr-movielens100k-context50-three-seed-results.md/.json;receiptcomplete.
  No pending GPU work, new inference/training/tuning/dispatch/retries/Telegram.


- Context50 replication authorized 2026-11-04 MSK: only seeds101/202, control
  and actor graph.1, four fresh generators, ordinary eager T5,30epochs on2T4.
  Config experiments/movielens100k_context50_seeds101_202.json; NEWprivate kernel
  carasik/rhpr-ml100k-context50-replication. Frozen original corresponding SIDs,
  exact original101/202 initializers, no AE/RQ/graph fitting. Same protocol as42:
  FP16AMP/physical128/effective256, unmaskedbeam20FP32, val5/10/15/20 then21-30,
  exact optimizer/scaler/RNG resume and old LR schedule. No extensions/HPO/new
  windows/more seeds/retries/hardware or batch fallbacks. Preserve42completed.
  Quota3.57h;expected2-2.5h elapsed;session timeout180min. SUBMITTED ONCE v1 RUNNING.
  Executable d8027d0d84d375659335680f63468423718c4a37;workflow41821006440
  succeeded submit-only. Tests958OK91skips;4native full-size backward probes OK.
  All197 embedded sources/config/SHA verified;no paused local TPU edits included.
  Receipt docs/gr-movielens100k-context50-replication-submission.json. No results
  yet;collect existing kernel after completion. Do not resubmit or extend.
  These pipeline seeds vary original frozen indices and generator initialization.
  Same previously examined holdouts; report val/test and all unfavorable results.


- Context50 COMPLETE, collected 2026-11-04 MSK. Private carasik/rhpr-ml100k-
  context50 v1; executable d79fdcee647729dfc1335e690aaf09b724439ab4. Two fresh
  seed42 predictors, frozen original indices, 30 epochs/10020 updates/14 checks,
  best epoch26 for both; 0 AMP skips/failures/invalid test paths/duplicates.
  L50 control/graph val .062088/.067638 (+8.94%); test .047234/.042117 (-10.83%).
  Graph50 vs40 val+2.50%, test-28.46%; control val-11.60%, test+3.50%.
  Graph validation rises20/30/40/50 on42, but test does not; one exploratory seed,
  no established longer-context gain. Fixed epochs20/25/30 also favor graph val.
  Conditional graph-control50 CIs cross0; graph50-40 test CI negative, conditional
  on fitted models only. No seed variance or repeated-holdout correction.
  Epochs130.30/122.37s, peak allocated2.400GiB, runner66.87min. Test known slots
  control38.93%/graph40.55%; repeats persist. Same eager/FP16/beam20 protocol.
  94 distinct prediction files rescored to1e-12; index/init/schedules/resume audited.
  Report docs/gr-movielens100k-context50-results.md/.json; receipt marked complete.
  No new GPU work or fits pending. Preserve old runs; no resubmit/extensions/HPO.


- Context50 authorized 2026-11-04: ordinary eager T5, seed42 only, control and
  actor graph .1, 2 fresh predictors on 2T4, 30 epochs, same protocol as L40.
  Config experiments/movielens100k_context50_seed42_30epochs.json; new private
  kernel carasik/rhpr-ml100k-context50. Original frozen seed42 SID indices and
  fresh original initialization; FP16 AMP, physical128/effective256, beam20
  FP32 unmasked eval. Val5/10/15/20 then21-30; exact resume and old LR schedule.
  Only context changes; no quantizer fitting/Flash/TPU/HPO/extra seeds/retries.
  Quota4.69h before run; expected ~1-1.5h elapsed, not measured. Preserve old
  completed runs. Report validation/test separately; one exploratory seed.
  SUBMITTED ONCE v1 RUNNING; source d79fdcee647729dfc1335e690aaf09b724439ab4.
  Workflow41809406443 succeeded submit-only;197 embedded sources/config verified.
  Tests957OK91skips;5native runner tests OK; no results yet, no resubmit.
  Receipt docs/gr-movielens100k-context50-submission.json. Collect after completion.



- Context40replication COMPLETE,collected2026-11-04MSK. Private carasik/rhpr-
  ml100k-context40-replication v1,e3308db026871ccec6590ba502b6cce8d65ebd0c,
  4newfits101/202/102.45min;reuse2L40seed42fits,notretrained. All6models
  30epochs/10020updates/14valchecks/0AMPskips/0invalidtest/0duplicates.
  L40test control/graph:42 .04564/.05887(+29%),101 .04862/.04759(-2.1%),
  202 .04656/.04812(+3.3%). Mean.04694/.05153(+9.8%)drivenby42;
  new101/202only+0.56%. Valmeancontrol.06789/graph.06596(-2.8%),graphwins1/3.
  ConditionaltestCI[-.00217,+.01133]andvalCIcross0;notseeduncertainty.
  Matched<=30epochval graph20->40positive3/3:+4.0/+8.5/+5.3%,mean+5.8%,
  CI[-.00380,+.01112]cross0;controlmean-1.34%. Descriptiveleadonly.
  L40testknownslotscontrol40.23%/graph40.28%;repetitionstill~40%.
  Newepochs96-100s;peakfitallocated1.84GiB,physical128unchanged. Fixedval
  epochs20/25/30graphmeanalsoworse. Selectedepochs101control25/graph28,
  202control26/graph28. Prior42graph30cap. No convergence/generalgraphclaim.
  102predictionfilesrescored1e-12;allcase/index/init/resumeproofsOK.
  Tests956OK91skips. Report docs/gr-movielens100k-context40-three-seed-results
  .md/.json;artifacts/movielens100k-context40-replication/analysis.
  NoGPUworkpending. Noextra inference/training/tuning/extensions/seeds/Telegram.



- Context40replication SUBMITTED ONCE,v1RUNNING,2026-11-03MSK. Private
  carasik/rhpr-ml100k-context40-replication;workflow41682670062successsubmit-only.
  Executablee3308db026871ccec6590ba502b6cce8d65ebd0c;197sources/exactconfig
  verified;onlyscopevalidatorchangedvs42. Fourfreshfits L40none/actor.1 x101/202,
  30epochs,originalperseedfrozenSIDs,FP16/eager/unmaskedbeam20FP32eval/2T4.
  Runtime ready;workersstartednone_L40_seed101 andactor_w01_L40_seed101;
  then202. Tests956OK91skips;4realconstructor/backward checksOK.
  Receipt docs/gr-movielens100k-context40-replication-submission.json and
  artifacts/movielens100k-context40-replication. Seed42completeunchanged.
  Noresultsyet;collectexistingrun aftercompletion. Noresubmit/extensions/HPO/
  morewindows/seeds/precision/hardwarefallback/Telegram.



- User authorized two more seeds2026-11-03 after context40signal. Prepare ONLY
  L40 control/actor.1 x101/202,4fresh predictors,30epochs,sameprotocolas42.
  Config experiments/movielens100k_context40_seeds101_202.json; doc
  gr-movielens100k-context40-replication.md. Original SIDperpipeline seed frozen;
  predictor initialization matches original101/202;noAE/RQorgraphretraining.
  Unmaskedbeam20/FP32eval,FP16AMP/eager,effective256/physical128. Prefix20val5,
  then21-30valeach/verifiedresume/oldplateauLR. NoHPO/newwindows/extensions.
  NEWprivate kernel carasik/rhpr-ml100k-context40-replication,2T4,quota6.42h.
  Expected~1.5-2h;seed42source results immutable. Notgenerator-only variance.



- Context30/40 COMPLETE,collected2026-11-03MSK,private carasik/rhpr-ml100k-
  context30-40 v1,sourcebfecf91e658d24415255b480102c9f94e09fcc59,86.08min.
  Fourfreshseed42predictors,frozenoriginalSID,30epochs/10020updates/14valchecks,
  0AMPskips/failures/invalidtest/duplicatetest. NoGPUworkpending.
  Matched<=30epochval control20/30/40=.06656/.06544/.07024;
  graph=.06345/.06461/.06599. Graph losesall3valwindows.
  TestL30control.04790/graph.04361;L40.04564/.05887(+29%graph),butONEseed,
  valsignopposite,no robustgainclaim. HistoricalL20graph test.04593isepoch31,
  NOT30budgetmatched(graphvalbest<=30is24);controltest.04530epoch26matches.
  Selectedepochs control30=27,graph30=27,control40=27,graph40=30(cap).
  TestrepeatsL30control43.34%/graph42.32%;L40 38.82/40.25%. Fixedvalepoch30
  directionalsofalls;repetitionnotremoved. MeanepochL30 66.7-69.4s,L40 92.9-
  97.8s;peakfit+valallocated1.33/1.84GiB,physical128unchanged.
  64distinctpredictionfilesrescored1e-12;allSID/hash/case/init/resumeproofsOK.
  Tests955OK91skips. Report docs/gr-movielens100k-context30-40-results.md/json,
  artifacts/movielens100k-context30-40/analysis. Noextensions/newruns/seeds.
  Do not inferhistorycausality:trainingchangesallusers;fixedleftpad+prepend
  usertoken changesuser-item relative distances whenmaxwindowchanges.



- Context30/40 SUBMITTED ONCE,v1RUNNING,2026-11-03MSK. Private
  carasik/rhpr-ml100k-context30-40; workflow41662734850success submit-only.
  Executablebfecf91e658d24415255b480102c9f94e09fcc59;197embeddedsourcefiles
  andexactconfigverified;2T4/private/nointernet. Runtime ready;workers started
  none_L30_seed42 andactor_w01_L30_seed42. Fourfits total, thenL40;seed42,
  max30epochs, noFA/no seenmask, oldfrozenSIDs/freshpredictors. Tests955OK91skip,
  native3OK,realinitializers/backwards4OK. Receiptdoc gr-movielens100k-context30-
  40-submission.json and artifacts/movielens100k-context30-40. Noresults yet.
  Do not resubmit/extend/tune/addseeds;collect thisexistingrun oncompletion.



- User authorized context training2026-11-03: NOFlashAttention, windows30and40;
  clarified seed42ONLY and30epochs. Four fresh TIGERgenerators: none/actor.1
  x30/40. Frozen original seed42SIDs, no AE/RQ fitting or predictor warmstart.
  Config experiments/movielens100k_context30_40_30epochs.json; module
  rhpr.gr.movie_context_training; doc gr-movielens100k-context30-40.md.
  SameFP16AMP/eager, effective256/physical128, unmasked beam20/FP32eval.
  Prefix20val5then21-30valeach withverifiedoptimizer/scaler/RNGresume;
  plateauLRfrom21; bestvalcheckpoint;30epoch cap isNOTconvergence evidence.
  Four realCPUconstructor/backward checks matchoriginal5f4a24fa… initializer.
  Original85470targets/943heldoutusers andlast20suffix allverifiedunchanged.
  CompareL20validationwithinfirst30;historicalL20testhaslongertrainingbudget.
  Private NEWkernel carasik/rhpr-ml100k-context30-40;2T4;quota7.86hbefore.
  Expected~1.5-2helapsed. Noextras/seeds/extensions/precision/hardwarefallback.



- Unseen serving COMPLETE2026-11-03,private carasik/rhpr-ml100k-unseen-serving v1,
  source8f0b782b387a5f464bdd514fa6ae5d10f735b764/workflow41649512819.
  6original fixed best checkpoints,0trainingupdates,2T4,runner76.44s.
  Known-past mask+beam100/refill256: test control.076682/graph.082005(+6.94%),
  val.108692/.103901(-4.41%). Test2/3wins;conditionalCI[-.00258,+.01324]cross0.
  All12x943outputs10uniquevalidunseen;13casesneeded256,0exactfallback.
  Parent32-user20beam anchors exact;allweightsunchanged;metricsrescored1e-12.
  Original unmasked results unchanged;new servingpolicy,old checkpoints/no
  reselection. No reliable overall graphgain established. No pendingGPUwork.
  Actorgraph reconstructed15998nodes/475238edges. Frequency/promotion rho~.02,
  partial aftermoviepop/cast~.03;strong repeatededges~.03test/~0val. Context
  normalizedaffinity promotion partialPearson-.019to+.031acrosssplits/seeds.
  No strong actorhub/PopBias explanation. No externalcelebrity data.
  Report docs/gr-movielens100k-seen-serving-results.md;summary JSON/replay
  scripts/analyze_actor_graph_bias.py committed. Preserve originalprotocol.



- User authorized unseen-item serving inference2026-11-03 for all6frozen actor
  full/control checkpoints and graph popularity/edge-strength analysis.
  Config experiments/movielens100k_actor_seen_serving.json; module
  rhpr.gr.movie_seen_serving; doc gr-movielens100k-seen-serving.md.
  No retraining/reselection. Beam100unmasked plus known-past eligible top10,
  refill256onlyshortusers, exact full-catalogue fallback onlyremaining shortages;
  preserve raw AR scoring, require no repeats, verify parent hashes and beam20
  replay anchor. Private NEWkernel rhpr-ml100k-unseen-serving; parent immutable.
  Tests952OK90skips; native raw-score/exact fallback check passed.
  Current /tmp caches from prior turns were wiped; rawsource remains onKaggle.



- Actor harm audit COMPLETE2026-11-03, CPU cached predictions only; no GPU.
  docs/gr-movielens100k-actor-harm.md, artifacts/movielens100k-actor-full/harm.
  Official unmasked metrics unchanged. Test known-seen top10 occupancy control
  41.35%/graph42.83%; retained-beam20 all-known-past filtering testNDCG
  .071584/.077037, val .104387/.100386; not new decoder/official scores.
  Only13/216 original harmed test user-seed cases become ties, none strict wins;
  repeats do not explain all graph harm. Filtered test CI crosses0,seed101loses.
  Actor-overlap promoted unseen wrong candidates test67.5vs63.4%, but val61.5vs
  61.6%; no validated harmful-edge gate. Popularity/hub mechanism unsupported.
  Known-past excludes embargoed timestamp siblings; targets verified not seen.
  Preserve original protocol; any future mask/refill evaluation separate.



- Actor subgroup audit COMPLETE2026-11-03,CPUonly/noinference/training.
  Full3seed andscreen2seed cachedpredictions;20selectedfilesrescored,
  exact943cases/hashes/IDsverified.104group rows/32contrasts retained.
  No group positiveonall3fullseeds onBOTHval/test. Director-overlap test
  n57 .08034->.09906(+23.3%,3/3but42neartie,25->33hitsacross171fits),
  valn54 .06204->.05596(-9.8%);oldscreen testmean+37.6%but1/2seeds only.
  Gainnotonesingletarget, butall3signbreaksremovingBatman&Robin;CIs cross0.
  Rare<=20trainingpositions testn84 .001709->.003897 isonly1->3hits across
  252fittedcases;screenvalopposite,nocoldstartclaim. Past>=100val+12%(3/3),
  test-2.7%;actualinputstilllast20. Graph-coherenthistory screenpositiveboth
  splits/2seeds,fulltestnegative3/3;no transferablerule. Directactoroverlap,
  repetition/support/castsize notstable. Secondaryfixedmedianbaselinebeamgap
  probe:graphlowconfidence test+6.5%,val-12.3%;anotherbaselineseedtest+10.3%,
  notgraph-specific. Hypotheticalgatevalnegative3/3;notselected/deployed.
  Only6testusersbetterall3and6worseall3,72bothsigns;examples/influence saved.
  AllconditionalCIs unadjusted/excludetrainingvariance;holdoutsalreadyused.
  Report docs/gr-movielens100k-actor-subgroups.md; artifacts/
  movielens100k-actor-full/subgroups links /tmp/actor-full-subgroups withall
  CSV/ranks/features/replays/plots. EvidenceZIP in/tmp. Tests949OK90skips.
  No newGPUwork/gates;preservepriorstudies/controls/pausedTPUedits.

- Full actor comparison COMPLETE,collected2026-11-03MSK. Private
  carasik/rhpr-ml100k-actor-full-training v1,6freshfits/3seeds/2T4,159.34min,
  nofailures/allES/nonecapped. Mean testNDCG10 control .04811±.00271,
  actor.1 .04722±.00500(-1.9%);seed42+1.4%,101-15.2%,202+9.1%(2/3wins).
  Valcontrol .07077/graph .06521(-7.9%,graphloses3/3). Shared-user conditional
  testCI[-.00714,+.00539] crosses0;no robustgain,equivalence orstableharm.
  Prior+7.4%screen notreplicated;freshAE/RQindicesalsochanged despite same
  initialAE/generatorweights/config/features/normalization. L1ARIvsoldscreen
  .235–.310;notjustlabelpermutation. Exactnumerical/runtimecauseunknown;
  generatorCEdiff alreadyepoch1. Cannotattributeflipsolelytolongtraining.
  Best/stop control42:26/46,101:31/51,202:34/54;graph31/51,48/68,29/49.
  All229predfilesrescored,372filesdownloaded;casehashes/actorinputs/norm/pins/
  AE-RQtransfer/prefixresume/LR/ES verified. AMPskipsgraph42=1,101=3,
  control202=1;testinvalidcontrol42=2,202=1,duplicates0. Tests949OK90skips.
  Report docs/gr-movielens100k-actor-full-results.md; artifacts/
  movielens100k-actor-full/analysis linksCSV/plots/replay/ranks in/tmp;
  numericalZIP in/tmp;largeweightsprivateKaggle. No pendingGPUwork and no
  automaticnewruns/extensions/retries/seeds/tuning/graphchanges/TPU/Telegram.
  Preserveallpriorstudies/controls/pausedTPUedits. Publicationpositivegraph
  claimnotestablished;previouslyexaminedholdout andtraininguncertaintyremain.

- Full actor comparison submitted ONCE,v1RUNNING,2026-11-03MSK. Private
  carasik/rhpr-ml100k-actor-full-training;6freshfits/3seeds/2T4.
  Workflow41535407288 successsubmit-only;executable
  70a0c16d65fab69df3d64742fee527975e12df70. Exactconfig/all195embedded
  sourcefilesverified;packageSHA
  d22ad4da3631a43efeb72f9b7d53701861b8230e05b5e5d7e7c4792c3c235595.
  Tests949OK90skips;newfullsuite2checks andlegacyactorsuite3checks native
  GRIDpassed;all6real-schema constructorpins(including202) verified.
  Frozenactorgraphhash/orderverified. Receipt
  docs/gr-movielens100k-actor-full-submission.json; artifacts/
  movielens100k-actor-full haspreflight/submissionaudit/quota. Samefullpolicy:
  fresh20prefix,val5 thenverifiedresume,last/Adam/scaler/allRNG,min30/max100/
  ES20/valeachnew/ReduceLRfrom21. .1frozeninAE+RQ,nonecontrol,seed42/101/202.
  Noresults;waitforusercompletion,collectexistingrun. Do notresubmit/extend/
  tune/newseeds/changegraph/TPU/Telegram. Preserveallprioroutputs/pausedTPU.

- User authorized full actor comparison2026-11-03. Config
  movielens100k_actor_full_2t4.json;doc gr-movielens100k-actor-full.md.
  SIXfresh fits: none vsnewactorweight.1 ×42/101/202. Coefficient selected
  on priorval, frozen. Samefeatures/graph15998nodes seed42, schemas/pins,
  cast-only64Dloss/normalization, AE3500fixed/RQ3000, .1inbothstages.
  Fullpriorprotocol: fresh20prefix,val5,verifiedlast/Adam/scaler/allRNG
  resume at20; thenvaleachnew,min30/max100/ES20, ReduceLRfrom21factor.5
  after5badchecks/floor1e-5/threshold1e-4. One finaltest,noneat20boundary.
  Screen30lastcheckpoints existbutconstantLRthrough30incompatiblewithprior
  fullschedulefrom21;all6fresh forcommonprotocol. SameTIGER4+4/d128/FP16.
  2T4/3trialseach,pairs42/101sameworker,202splitforbalance. Private target
  carasik/rhpr-ml100k-actor-full-training. NoHPO/extensions/retries/extraseeds/
  graphchanges/TPU/Telegram. Preserveallpriorcontrols/pausedTPUedits. Old
  holdout alreadyexamined;notindependentconfirmation orDeepWalk-vsidentity.

- Actor-only graph screen COMPLETE,collected2026-11-03MSK. Private
  carasik/rhpr-ml100k-actor-screen v1,10fits/2seeds/2T4,157.86min,nofailures,
  exactly30epochs/6valchecks/constantLR/zeroAMP. Mean val/testNDCG10:
  none .05695/.04223;oldsharedcast1 .06021/.03957;newactor1 .06047/.04308;
  newactor.3 .06117/.04247;newactor.1 .06325/.04535. Weight.1 winsval2/2,
  testvsnone+7.4%(2/2;+13.7/+1.4%),conditionaluserCI[-.00450,+.01083]
  crosses0;notastablebenefit/optimalityclaim. .1vsnew1testpositive1/2 only.
  Selectedepochs .1 30/25;fixed30 is a screen,notconvergence. RawotherheadMSE
  improves1->.1 onbothseeds,peopleworsens;notaprovenNDCGmechanism. Actor1
  highseedspreadalsowithoutdirector edges;priorSDnotcausalproof.
  All70predfilesrescored,260filesdownloaded;exactcases/hashes/normalization/
  actorprovenance/pins/AE-RQtransfer/weights/schedule verified. Oneinvalidtest
  actor.3seed42,others0;duplicates0. Report
  docs/gr-movielens100k-actor-screen-results.md; artifacts/
  movielens100k-actor-screen/analysis links fullCSV/plots/replay/ranks in/tmp;
  numericalZIP in/tmp;largeweightsprivateKaggle. No pendingGPUwork; do not
  automaticallyextend/tune/addseeds/weights/retrainGraph/TPU/Telegram.
  Tests947OK89skips. Preserveallpriorcontrols/pausedTPUedits. Primarycandidate waschosenonval,
  testdescriptive,alreadyexaminedholdout; noDeepWalkoveridentityclaim.

- Actor-only graph screen submitted ONCE,v1QUEUED,2026-11-02. Private
  carasik/rhpr-ml100k-actor-screen;10fits/2seeds/2T4,5arms×42/101.
  Workflow41481526966 success submit-only;executable
  5858df9f4682a15fdc740d3717469c8d4c6e22c5. Exactconfig/all194embedded
  sourcefiles verified;packageSHA
  be2b6eb81133a895d200f9cc363d9eb48e1058fabaf362e553b86c66d07f72eb.
  Newprivate actorDeepWalkdatasetREADY;15998nodes, same1273items, exactnode
  set/filmmeanpooling/hash/order verified. Otherfrozenmodalitiesbitwiseequal.
  Tests947OK89skips;3nativeactor/3legacyrolechecks passed. Receipt
  docs/gr-movielens100k-actor-screen-submission.json; artifacts/
  movielens100k-actor-screen haspreflight/graphprovenance/submissionaudit.
  Fixed30epochs,val5,constantLR,bestval; weights1/.3/.1 inAE+RQ, newactorgraph
  plusnopeople/oldsharedcast controls. Noresults;waitforusercompletion,
  collectexistingrun. No resubmission/extensions/tuning/newseeds/TPU/Telegram.
  SDKstatusdefaultQUEUED=0 is omitted byMCP whenqueued. Preservepriorstudies.

- User authorized actor-only graph/loss screening,2026-11-02. Config
  movielens100k_actor_screen_2t4.json; doc gr-movielens100k-actor-screen.md.
  Two pipeline seeds42/101 ×5arms: nopeople,old sharedgraph cast(weight1),
  new actor-only graph weights1/.3/.1. FIXED30epochs,val5/10/15/20/25/30,
  constantLR/no scheduler, bestval selection, one test; primary tuning uses
  meanval only. AE3500fixed/RQ3000, coefficientinbothstages, otherweights1;
  selectedcast64Dnormalized/zero-padded/loss only64, same nativepins/schema.
  NewDeepWalk oncastONLY:15998people/same1273movies, samegensim4.4 SGNS/walk
  settings/seed42. CPU graphprepared once, savednewprivate dataset
  carasik/rhpr-ml100k-actor-deepwalk, hashes/sourceorderverified. No interactions.
  Tenfits/2T4, sameGPU perseed all5arms; private kernel target
  carasik/rhpr-ml100k-actor-screen. No retries/extensions/newseeds/weights/TPU/
  Telegram. Old fulltrainingcontrol notmatched to30epoch screen. Preserve all
  prioroutputs/controls andpausedTPUedits. DirectorSD3seeds is not causal
  proof of instability. No graph-identity advantage claimed.

- People-role ablation COMPLETE,collected2026-11-02MSK. Private
  carasik/rhpr-ml100k-people-roles v1;9fits/3seeds/2T4,227.51min,allES,
  nonecapped/nofailures. Mean testNDCG10 nopeople .04703±.00269,
  cast .04525±.00234(-3.8%),directors .04182±.00570(-11.1%); each wins1/3.
  Conditional paired-userCIs cross0; no stablebenefit/harm established.
  Valcast .06845 vscontrol .06411,positive3/3. Same-director groupval(n54)
  control .05860/director .08275,buttest(n57) .09735/.08705; subgroup
  observation doesnottransfer. Fixed3500AE: historicalseed42samecodes/AE,
  differentgenerator;101/202oldAE4000/5000yielddifferentindices,usefreshcontrol.
  All291predfilesrescored,494files/2pagesdownloaded;roleloss/normalization,
  pins,AE/RQtransfer/resume andLR/ES verified. AMPskips1eachcast42/cast101/
  director101;invalidtest12none202/1director202,duplicates0. Tests944OK88skips.
  Report docs/gr-movielens100k-people-roles-results.md; compactartifacts/
  movielens100k-people-roles/analysis; fullreplay/CSV/plots/ranks /tmp/movie-roles-analysis,
  numericalZIP in/tmp; largeweightsremainKaggle. No pendingGPUwork. No
  autoextensions/tuning/newseeds/retrainingDeepWalk/TPU/Telegram. Preserve
  priorstudies/controls andpausedTPUedits.

- People-role ablation submitted ONCE,v1RUNNING,2026-11-02. Private
  carasik/rhpr-ml100k-people-roles;9freshfits/3seeds/2T4. Workflow41441646043
  success submit-only; executable682a23dabed6803d3340607f6ecd7924483ce3ba.
  Exactconfig/all193embeddedsourcefiles verified;packageSHA
  8e130ae812b890ccc7a8adc16b60d60956c5ff8c51ae0cafe63401458172b6ba.
  Receipt docs/gr-movielens100k-people-roles-submission.json; artifacts/
  movielens100k-people-roles haspreflight/submissionaudit/quota.
  Tests944OK88skips,3native rolechecks(incltinyAE/RQ/fullresume) and2legacy
  graph-losschecks passed. Sameinitialneuralpins3seeds and real1273film role
  isolation verified. Noresults yet;waitforuser,collectexistingrun. Do not
  resubmit/extend/tune/addseeds/retrainDeepWalk/TPU/Telegram. Preserve all
  oldoutputs/controls andpausedTPUedits.

- User authorized people-role ablation on2026-11-02: nine fresh fits,
  seeds42/101/202 × nopeople/cast-only/directors-only. Config
  movielens100k_people_roles_2t4.json; doc gr-movielens100k-people-roles.md.
  Frozen old DeepWalk128D shared-table vectors; selected64 role coordinates
  independently catalogue-normalized and zero-padded to original128D schema;
  loss only on selected64, all active modality weights1. Control omitspeople.
  Same otherfeatures/split/cases/nativepins. AE3500fixed/RQ3000 and full
  TIGER policy: prefix20val5, verifiedresume, min30/cap100/ES20/valeachnew,
  unchangedReduceLRfrom21; one finaltest. 2T4,private
  carasik/rhpr-ml100k-people-roles. No HPO/retries/extraseeds/TPU/Telegram.
  Prior subgroup audit: same-director fullval .05090->.07763(n54),
  test .07088->.09863(n57), but earlyval opposite and conditionalCIs cross0.
  Sharedcast no advantage; groups exploratory. This tests role readout, not
  typed graph training or a proven DeepWalk advantage over identity.
  Preserve all previous outputs, controls and paused TPU edits.

- Graph loss weighting pilot COMPLETE, collected 2026-11-02 MSK. Private
  carasik/rhpr-ml100k-graph-loss-weight v1, two seed42 fits, body46.97min,
  no failures, both early-stopped. Weight1 vs0.1: valNDCG10 .06786/.06601;
  testNDCG10 .04554/.04202 (-7.7%); Recall10 .09226/.08590. Best27/20,
  stop47/40. Conditional paired-user delta95%CI[-.01534,+.00757],63better/
  64worse/816same; excludes training variance. One paired seed, no stable
  benefit/harm established. Both AE3500fixed/RQ3000, identical initial pins,
  zeroAMPskips. Text/poster/plot raw MSE all improve with weight0.1; graph MSE
  worsens .79349->.88231. Text-neighbor firstSID4.10->4.74%, graph3.86->1.75%.
  Representation balance changes but no downstream gain. Both1273unique tuples.
  IMPORTANTfresh weight1 control exactly reproduces old graph42 AE neural
  state/codes/fullSID/reconstruction and generator initial neural hash, but
  generator train loss differs already epoch1 (9.2839231 vs9.2836568); test
  .04075historical -> .04554fresh. Numerical/runtimecauseunidentified. Use
  fresh pairedcontrol, not historical substitution; generator repeat variation
  exceeds treatment gap. Independentlyrescored57prediction files, same943cases,
  verified losses/features/AEtransfer/index/prefix/resume, replayed LR/ES.
  TestinvalidSID0control/1weak among18860slots; no duplicates. Tests941OK87skips.
  Report docs/gr-movielens100k-graph-loss-weight-results.md; artifacts/
  movielens100k-graph-loss-weight/analysis hasplots/CSV/ranks/replay. Full20.1MB
  evidenceZIP in/tmp; largeweights stay private Kaggle. No pendingGPUwork;
  do not autoextend/tune/addweights/seeds/TPU/Telegram. Preserve oldstudies,
  controls and paused TPU edits.

- Graph loss weighting pilot submitted ONCE, v1 RUNNING, 2026-11-02 MSK.
  Private carasik/rhpr-ml100k-graph-loss-weight; two fresh seed42 pipelines,
  people-reconstruction weights1/.1 in AE+RQ, all other weights1. Two T4s.
  Executable35b75d22360eada2053f59177acd18f928fbd867; Actions41427695618
  success submit-only. Exactconfig/all192embedded source files verified,
  packageSHA e9f19fc5da5d2ea2940af7128a7fdda181ff9e8e3b7327bef463334098002502.
  Receipt docs/gr-movielens100k-graph-loss-weight-submission.json;
  artifacts/movielens100k-graph-loss-weight has preflight and submission audits.
  Both fixed3500AE/nativeRQ3000, same frozenfeatures and native initial pins;
  TIGER prefix20val5 then verified resume, min30/cap100/ES20, unchanged LRpolicy.
  Single finaltest; raw perheadMSE+weighted totals logged. Tests941OK87skips,
  2new native+9legacy native checks passed. No results yet; wait for user,
  collect existing run. Do not resubmit/extend/tune/addweights/seeds/TPU/Telegram.
  Preserve all previous notebooks/results and paused TPU edits.

- User authorized graph reconstruction weighting pilot, 2026-11-02 MSK.
  One private2T4 session, two fresh seed42 pipelines, people-loss coefficient1
  versus0.1; other3reconstruction weights1. Apply in both continuous AE and RQ;
  no input scaling, changed graph/features/split or quantization auxiliary.
  Config movielens100k_graph_loss_weight_2t4.json; doc
  gr-movielens100k-graph-loss-weight.md. Both fixed3500AE updates (old42actual
  length), nativeRQ3000, same neural initialization/hash-pinned frozen inputs.
  Fresh control avoids adaptive warm-up duration confound; one-seed pilot only.
  TIGER full policy: first20epochs val5/10/15/20, verified last/Adam/scaler/RNG
  resume in-place at20, then val each epoch, min30/cap100/ES20 and same LR
  schedule from21. No test at boundary, one selected test. Preserve raw and
  weighted head losses, codes/books, CE, curves, prefix/resume provenance.
  Tests941OK87skips; new native2 and legacy multimodal9 all passed; exact
  decoder graph gradient0.1 scaling/text unchanged, all native init pins match.
  Private target carasik/rhpr-ml100k-graph-loss-weight; ~45–75min+setup/queue
  if similar stopping, near-cap ~1.5–2h. One trial per GPU. No weight sweep,
  extra seeds/retries/extensions/TPU/Telegram; preserve all prior studies and
  paused TPU edits. CPU diagnostic evidence from preceding turn is local at
  artifacts/movielens100k-people-graph-audit/followup-2026-11-02: case hashes
  reconstructed exactly; graph-neighbor firstSID agreement1.86->3.92%, text
  5.31->4.54%; text/poster recon worsens3/3. Observational, not causal proof.

- Full-training replication COMPLETE, collected 2026-11-02 MSK. Extra seeds101/202
  private carasik/rhpr-ml100k-full-training-extra-seeds v1: all8fits successful,
  body109.43min. Combined with retained42 pilot:12fits, all early-stopped,
  none capped. Mean testNDCG10±sampleSD: nopeople .04672±.00631;
  graph .04260±.00380; names .04469±.00046; SASRec .05214±.00343.
  SASRec beats each TIGER on3/3; nopeople mean-10.4% vsSAS. Relative change
  from selected20epoch baselines: nopeople+8.9%(2/3),graph-1.5%(1/3),
  names+18.5%(3/3),SAS-.9%(two unchanged). Graph vsnopeople -23.3/-1.1/+1.1%
  across42/101/202: negative mean driven by42, no stable graph advantage.
  Graph selectedVAL beats nopeople3/3 despite opposite mean TEST ordering;
  943target holdouts and3pipeline seeds limit conclusions. Seed42near-tie
  withSAS not replicated. Best TIGERepochs25–41; stop45–61; SASbest10/5/25,
  stop30/30/45. Four new TIGER selections follow LR drop; do not generalize
  seed42's pre-drop peak. TrainCE falls/valCE rises afterselectedpoint.
  Independentlyrescored396prediction files, matching cases/indices/prefix
  hashes/resumeAdam-scaler-RNG and replayed all LR/ES decisions. AMPskips:
  nopeople101=1/nopeople202=3, others0; counts verified, no fallback.
  Report docs/gr-movielens100k-full-training-extra-seeds-results.md and
  three-seed-metrics.csv; artifacts/movielens100k-full-training-extra-seeds/
  analysis has metrics/curves/verification/replay. Full50.8MBnumericalZIP in/tmp,
  largeweights remain private Kaggle. Tests939OK86skips. No pendingGPUwork,
  do not autoextend/tune/addseeds/retrain indices/TPU/Telegram; preserve prior
  studies, seed42 outputs and paused TPU edits.

- Extra-seed full-training replication submitted ONCE, v1 RUNNING, 2026-11-02.
  Private carasik/rhpr-ml100k-full-training-extra-seeds; eight continuations,
  seeds101/202, 2T4. Executable8e1819a6137aaae30e43a4115cb6a4c2aae1b819;
  Actions41406980487 success submit-only; exact config and all191 embedded
  source files verified, package SHA256
  bf97caa8609ff46e0e76099c87c65f5fa743384b1f783852b40c7764a961ae30.
  Receipt docs/gr-movielens100k-full-training-extra-seeds-submission.json;
  artifacts/movielens100k-full-training-extra-seeds has source/initialization/
  submission audits. All939 tests OK86skips;4 targeted/native passed.
  Frozen original epoch20/6680 checkpoints, min30/cap100/ES20, val each new
  epoch and unchanged pilot scheduler. No results yet. Wait for user completion;
  collect existing run, do not resubmit/extend/tune/addseeds/retrainindices/
  TPU/Telegram. Preserve seed42 output, all parent studies and paused TPU edits.

- User authorized full-training replication on existing seeds101/202, 2026-11-02.
  Eight frozen-index continuations: nopeople/graph/names TIGER + SASRec per seed.
  Config movielens100k_full_training_extra_seeds_2t4.json; doc
  gr-movielens100k-full-training-extra-seeds.md. Resume original epoch20/6680
  checkpoints with Adam/scaler/RNG/prior best/history; no encoder/AE/RQ/index
  retraining. Same seed42 pilot policy: min30/cap100, val every new epoch,
  ES20, LR halve after5 bad checks from21, floor1e-5/threshold1e-4.
  Same data/cases/architectures/batches/precision; 2T4, 3TIGER+1SAS per worker.
  Source120files/642053025bytes verified; per-trial seed/lineage and constructor
  initialization checked. Compare init within predictor+seed, not across seeds.
  Private target carasik/rhpr-ml100k-full-training-extra-seeds, one new session,
  ~1.5–2h plus setup/queue if similar stopping; later bests may take longer.
  Combine with fixed seed42 results for three-seed report, not test-selected
  epochs/seeds. No HPO/retries/additional seeds/TPU/Telegram; preserve old runs
  and paused TPU changes. Submission receipt will record exact commit/status.

- Full-training pilot COMPLETE, collected 2026-11-02 MSK. Private
  carasik/rhpr-ml100k-full-training-seed42 v1; executable ce0efd6d6816a24d5710d5baeda706abc3d8e8cb.
  Four frozen-index continuations, seed42 only, body48.23min, all early-stopped.
  Test NDCG10 old-selected -> new-selected: nopeople .03940 -> .05315 (+34.9%);
  graph .03605 -> .04075 (+13.0%); names .03891 -> .04520 (+16.2%);
  SASRec .05411 unchanged. All TIGER best27/stop47; SAS best10/stop30.
  TIGER peak27 PRECEDES first LR drop after32; longer training and denser
  validation changed together. Later train CE falls while val CE rises.
  Nopeople remains best TIGER on this seed, 1.8% below SAS; NOT a multi-seed
  equivalence claim or evidence graphs are generally harmful. Val/test rank
  differently. Independently rescored115 prediction files, same943cases;
  frozen indices/source prefix hashes/resume Adam-scaler-RNG verified;
  scheduler/ES replayed, zeroAMPskips. Report
  docs/gr-movielens100k-full-training-results.md; local artifacts/
  movielens100k-full-training-seed42/analysis has curves/CSV/ranks/replay.
  Numerical evidence ZIP in/tmp; large weights remain private Kaggle.
  No pendingGPUwork. Do not autoextend/tune/addseeds/retrain indices/TPU/Telegram;
  preserve prior results and paused TPU edits.

- Fulltrainingpilot submittedONCE,v1RUNNING,2026-11-02MSK. Private
  carasik/rhpr-ml100k-full-training-seed42;4continuations/seed42/2T4,
  nopeople/graph/namesTIGER+SASRec. Resumeepoch20/6680updates,frozenindices.
  Executablece0efd6d6816a24d5710d5baeda706abc3d8e8cb;Actions41363415397success;
  exactconfig/all191embeddedfiles matchcommit,packageSHA
  f961db92e96598f5b7433b7a2a5284a8e52133e5f26e7d348634a4e9ee80bd29.
  Receipt docs/gr-movielens100k-full-training-submission.json;localartifacts/
  movielens100k-full-training-seed42/source-audit.json. Min30/cap100,ES20,
  valeachnewepoch,LRhalveafter5badchecksfrom21,teacherforcedvalCEperlevel.
  Source64files/321MBverified in/tmp/movie-continuation-input. Noresults yet.
  Waitforusercompletion/status;collectexistingrun,do notresubmit/extend/tune,
  changeindices/retrainAE-RQ/addseeds/TPU/Telegram. Preserveoldoutputs/pausedTPU.

- User authorized seed42full-training pilot,2026-11-02MSK. Config
  movielens100k_full_training_seed42_2t4.json;doc gr-movielens100k-full-training.md.
  Fourcontinuations:nopeople/graph/namesTIGER+SASRec fromepoch20/6680updates;
  frozenindices/encoders/RQ,sourceartifactsSHA-pinned(64files,321MB verified).
  Resume lastmodel/Adam/scaler/CPU-CUDA-Python-NumPyRNG/priorbest/history,
  neverweights-onlyfallback. Minimum30,totalcap100;valeverynewepoch;LR.5after
  5failedchecks,start21,min1e-5,absthreshold1e-4;ES20epochs sincebestrawNDCG10.
  Preservehistorical5/10/15/20validation,lognewCEperSIDlevel+LR;onefinaltest.
  Newcheckpointskeep scheduler/casehashes. Tests937OK86skips,targeted/native8OK,
  including bitwiseCPUresume/dropout/Adam/RNG/scheduler andnativeGRID4headCE.
  Newprivatekernel target carasik/rhpr-ml100k-full-training-seed42,2T4,
  ~2–2.5h+setup/queue worstcase,ESmaystopsooner. No extra seeds/HPO/retries,
  indexchanges/AE-RQretraining/TPU/Telegram. Preserveoldoutputs/pausedTPUedits.

- Names/SASRec controls COMPLETE,collected2026-11-02MSK. Private
  carasik/rhpr-ml100k-names-sasrec-controls v1;source8c9200133eee6473776836e3b5344e02446c05a6;
  6fits/20epochs/1709400cases/6680updates/zeroAMPskips,body48.29min.
  Independentlyrescored90savedfiles(30new+60matchedreferences),same943users,
  targets/split,seedpinnedinitialweights. Realcapacity4.609536M SASRec vs4.590464M
  TIGER(+.415%);FLOPs .18737G/.42737G,NOTequalcompute.
  Mean testNDCG10: nopeoplemultimodalTIGER.04292;graphpeople.04326;
  namesintext.03772±.00357;SASRecID.05261±.00261. Recall10respectively
  .09084/.08766/.08059/.11099. Namesvsnopeople-12.1%(negative3/3),vsgraph-12.8%
  (negative2/3). SASRecvsgraph+21.6%(positive3/3),vsnopeople+22.6%(3/3),
  vsnames+39.5%(3/3),vsplaintextRQ.04143+27.0%(3/3). Meanselectedval
  SASRec.06463/names.05928;bestSASepochs42:10,101:5,202:10. Untunedbaseline;
  no universalGRvsdenseclaim. Allconditionalpaired-userintervalsforSASgains
  positive,butexclude seed/runtimeuncertainty andmultiplicity.
  EpochsecondsSAS21.45/TIGER50.75. FP32residentinferenceB1meansofseedmedians
  SAS8.32/TIGER43.51ms;B32 SAS8.86/TIGER69.16ms(perwholebatch). Same32real20item
  cases,warm10/sample50;SASexhaustive1273itemtop20 vsTIGERbeam20. Tinycatalogue
  makesdensescoringcheap;notaproduction-sizeefficiencyconclusion.
  Namesvectorsbitwiseequaloldstage07T5,poster/plot/masks unchanged,graphinactive;
  all3AEtransfer/freshoptimizer/initiallatent auditsverified. UniqueSIDtriples
  1272–1273/firstcodes240–254,sooccupancydoesnotguaranteeNDCG.
  Report docs/gr-movielens100k-names-sasrec-controls-results.md;artifacts/
  movielens100k-names-sasrec-controls/analysis hasCSV/plots/ranks/verification.
  Rawoutputs /tmp/movie-controls-completed;17MBbundle
  /tmp/movielens100k-names-sasrec-controls-numerical-evidence.zip;
  largecheckpointsstayprivateKaggle. No pendingGPUwork;noautotuning/extensions,
  extraseeds/TPU/Telegram;preserveoldoutputsandpausedTPUedits.

- Names/SASRec controls submitted ONCE,v1RUNNING,2026-11-01. Private
  carasik/rhpr-ml100k-names-sasrec-controls,6freshfits/3seeds/2T4.
  Executable8c9200133eee6473776836e3b5344e02446c05a6;Actions41350345747success,
  exactconfig/all189embeddedfiles verified;packageSHA
  b4063d809c548c84d2df8e08fc0ae56d469c520d9b960e11a948f8d721d6e9bb.
  Receipt docs/gr-movielens100k-names-sasrec-controls-submission.json;
  localartifacts/movielens100k-names-sasrec-controls contains sourceaudit and
  capacitypreflight. NamesembeddedinT5text,poster/plotretained,graphinactive;
  SASRecIDonly4.609536MactiveweightsvsTIGER4.590464M(+.415%),NOTequalFLOPs.
  20epochs,val5/10/15/20,fullcataloguecasesmatched. No results collected yet.
  Waitforusercompletion/status,collectexistingrun;do notresubmit/extend/tune,
  addseeds/TPU/Telegram. Preserveoldoutputs andpausedTPUedits.

- User authorized names-in-text multimodal TIGER + item-ID SASRec controls,
  2026-11-01. Config movielens100k_names_sasrec_controls_2t4.json;see doc.
  Six freshfits42/101/202×2predictors,balanced2T4. TIGERtextincludescast/directors
  viaexactpriorstage07T5vectors(max1587/no truncation),poster+MPNetplot unchanged;
  graphpeople inactive. Mount oldprivatetextladder+multimodaloutputs,hash+order
  verified,no newencoding/upload. SameAEup-to5000/nativeRQ3000/TIGER20epochs.
  SASRec existingmodel7layers/d256/heads4/FFN768/dropout.1,IDonly,tieditemmatrix,
  fullcatalogCE/noPADlogit/no samplednegatives. Same20itemstrictpastcases,
  85470targets/epoch,20epochs,batch256/128,Adam.001,FP16train/FP32eval,
  val5/10/15/20,one selectedtest. Preserve oldmodels andruntimepolicy.
  Realbackwardcapacity(nonembeddingactive) TIGER4590464/SASRec4609536(+.415%);
  total4977920/4940800. Gatewithin1%. ProfileB1forward20validitems inclheads:
  .42737G/.18737G supportedFLOPs;PARAMETERcapacitymatched,NOTequalFLOPs/time.
  Save B1/B32 servinglatency(10warm/50samples),samefullhistorytraincases;
  commontrainerhooksforfullcatalogSASRec,IDpredictionfilesdistinctfromSIDfiles.
  Groupinitializationbypredictor+seed,notacrossdifferentarchitectures.
  Tests932OK85skips;3targeted/nativeOKinclboth20epochcontrols,capacity,masks,
  causalityandnamesreplacement. Real1273film replacementarrays verified.
  Newprivatekernel target carasik/rhpr-ml100k-names-sasrec-controls,
  ~40–60min+setup/queue. No extraHPO/seeds/retries/TPU/Telegram.

- MultimodalAE ladder COMPLETE,collected2026-11-01. Private
  carasik/rhpr-ml100k-multimodal-ae v1;sourceba160aac3155831e06286b6e6924eb720b82dfc2,
  all12fits/20epochs/6680updates/1709400cases/zeroAMPskips;body121.16min.
  Independentlyrescored140predictionfiles(60new+60direct+15plain+5pilot),same943
  users/targets/split/normalization/frozenvectors/initialRQ+generatorweights.
  Direct->AE mean testNDCG10: text.03388->.04016(+18.5%,3/3positive),
  poster.03982->.04127(+3.6%,2/3),description.03760->.04292(+14.1%,2/3),
  people.03773->.04326(+14.7%,2/3). AEall vstext+7.7%(2/3),vsplain.04143+4.4%(2/3),
  no stablemodalitysuperiority;conditionalCIs exclude seed/runtimevariation,
  most cross0. AEvalmeans.05684/.06171/.05925/.05676(bestposter),oppositetesttrend.
  FullmeanRecall10.08766 equalsplain;description.09084 highest. Firstbook228–256,
  semantictriples1270–1273;allfullSIDsstillidentify1273. AE3500–5000/56–105sec,
  all spectralplateau;RQ92–115sec;TIGERmean49.87–50.71sec/epoch.
  IMPORTANTrepeattext42:AEneuralstate/latents/RQcurves/books/SIDs bitwiseidentical
  topilot,andgeneratorinitialweights/config same,buttraininglossdiffersbyepoch1;
  selectedepoch15->20,test.04210->.03762. Divergencelocalized togeneratortraining,
  exactnumerical/runtimecauseunknown;seeddoesnotensurebitwisedeterminism.
  Do not substitute betterpilotortreatitas4thseed. All12transfer/freshoptimizer/
  initiallatent audits pass. ExtraAEcompute/init/stagemodesconfounded.
  See docs/gr-movielens100k-multimodal-ae-results.md;artifacts/
  movielens100k-multimodal-ae/analysis hasreports/CSV/plots/ranks/replayscript.
  Rawdownloadsin/tmp/ml100k-modal-ae-completed;14MBnumericalbundlein
  /tmp/movielens100k-multimodal-ae-numerical-evidence.zip (workspace4MBfree).
  LargeweightsremainprivateKaggle. No pendingGPUwork;noautoextensions/tuning,
  extraseeds/TPU/Telegram;preservepriorresultsandpausedTPUedits.

- MultimodalAE ladder submitted ONCE,v1RUNNING,2026-11-01. Private
  carasik/rhpr-ml100k-multimodal-ae;12freshfits/42,101,202×4stages;2T4.
  Executableba160aac3155831e06286b6e6924eb720b82dfc2;Actions41331992852success,
  exactconfig+all187embeddedfiles verified. PackageSHA
  2b8f7d428c7162a00a638597e435ac6738af579cff1155e3078aefa5b3b91ca8.
  Receipt docs/gr-movielens100k-multimodal-ae-submission.json;localartifacts/
  movielens100k-multimodal-ae/source-audit.json. No qualityresults yet.
  Waitforusercompletion/status,collectexistingrun;do notresubmit/extend/tune,
  addseeds,TPUorTelegram. Keepdirectladder/pilotoutputs/pausedTPUedits intact.

- User authorized full multimodal ladder with two-stage quantizers,2026-11-01.
  Config movielens100k_multimodal_ae_2t4.json; see corresponding doc. Twelvefresh
  fits42/101/202×4cumulative(text/poster/MPNetdescription/DeepWalkpeople),2T4,
  sixfits/worker. Seed42text repeated inuniformsuite;priorpilot keptintact.
  Exactoldfrozenvectors/masks/order/split/cases/generator+quantizerinitialweights;
  hashverifiedoldprivatenotebookoutputmounted,noembeddingrecompute/upload.
  SameAErecipe as pilot:cap5000/probe500/min2000/patience4spectralplateau,
  full neural+BNtransfer/nativeKMeans/freshAdagradRQ3000;TIGER20epochs,
  val5/10/15/20,one selectedtest. Per-seed initialQhashes explicitly pinned.
  Tests929OK84skips,native9OKinclall12tiny stagedquantizers,3realinitialQhashes,
  missingplotmask,transfer/freshoptimizer/latents andtiny20epochTIGER.
  Newprivatekernel target carasik/rhpr-ml100k-multimodal-ae,~2–2.5h+setup/queue.
  No retries/tuning/extraseeds/TPU/Telegram;oldnotebooks/pausedTPU preserved.

- Text AE warm-up COMPLETE,collected2026-11-01. Privatecarasik/rhpr-ml100k-text-ae-warmup
  v1/source444ba8238b2e244b498c4bb23ada89c769723b63;body20.25min. One seed42,
  text-only,sameheadedarchitecture/frozenvectors/normalization/initialRQ+generator,
  same943users/targets/split;all15predictionfiles(3models×5) independentlyrescored.
  All20epochs/1709400examples/6680updates/zeroAMPskips. AE4500steps/69.64sec,
  spectralplateau;nativeRQ3000Adagrad91.49sec;TIGERmean50.73sec/epoch vs50.95direct.
  SelectedvalNDCG10 direct.05402 -> AE.06102(+13.0%),plain.06900. Selectedtest
  .03438 -> .04210(+22.5%),plain.04341(-3.0%vsplain). Recall10.06999 -> .09014.
  Bestepoch15. Firstbook15 ->252codes;semantictriples1025 ->1273(no collisions),
  finalused[252,238,231],textMSE.83511 ->.72496. AEbooksuntouched/neural+BNtransfer,
  freshRQoptimizer andinitiallatents(all1273,maxdiff0) verified. ExtraAEcompute/
  initialization/stagemodesconfounded;not proofparametercountoroccupancycausedgain.
  One seed,conditionalpaireduser95%CI delta[-.00235,.01794]crosses0.
  See docs/gr-movielens100k-text-ae-warmup-results.md;localartifacts/
  movielens100k-text-ae-warmup/analysis(report/CSV/plots/numerical-evidence.zip).
  LargeweightsstayprivateKaggle. No pendingGPUwork,do notresubmit/extend/tune,
  addseeds,TPUorTelegramwithoutnewscope;preservepriorresultsandpausedTPUedits.

- Text AE warm-up submitted ONCE, KaggleRUNNING,2026-11-01. Private
  carasik/rhpr-ml100k-text-ae-warmup v1;source444ba8238b2e244b498c4bb23ada89c769723b63,
  Actions41328218657successsubmit-only;exactconfig/all187embeddedfiles verified.
  One text-only seed42 AE->RQ->20epochTIGER,oldprivate notebook outputs reused.
  Receipt docs/gr-movielens100k-text-ae-warmup-submission.json;local audit in
  artifacts/movielens100k-text-ae-warmup. No quality results yet. Waitforuser,
  collectexistingrun;do notresubmit/extend/tune/addseeds/TPU/Telegram.

- Text-only seed42 continuous-AE experiment authorized and implemented,2026-11-01.
  Config movielens100k_text_ae_warmup.json; native continuous AE->RQ-VAE->TIGER.
  One cuda:0 worker/trial, frozen prior notebook output (hash verified), no new
  feature upload. See docs/gr-movielens100k-text-ae-warmup.md. Source
  444ba8238b2e244b498c4bb23ada89c769723b63; Actions41328218657 dispatched once.
  Do not duplicate; pending submission/status receipt. No retries/tuning/extra
  seeds/TPU/Telegram. Tests927OK83skips plus native7OK. Preserve paused TPU edits.



- MultimodalMPNetv2 COMPLETE,2026-11-01;12/12fits20epochs,no failures,
  sourceaddd3c326dbc55ddf18d2dad2c832bf43000a8e0,117.50minbody. Collected
  rawreports/sharedfeatures;60predictionfiles+3plaincontrols independently
  rescored,same943cases/split,1709400examples/6680updates/zeroAMPskips each.
  Generatorinitialhashpinned;RQinitialparams matchacrossstages withinseed;
  fixedfeatures+normalizationidentical;base text exactlyoldstage5T5vectors.
  Mean testNDCG10±SD:plainRQsame text.04143±.00335;headedtext.03388±.00264;
  +poster.03982±.00163;+MPNetdescription.03760±.00599;+people.03773±.00171.
  Posterincrement+17.5%positive3/3;plot-5.6%positive1/3;people+0.3%positive1/3.
  Headedtext-18.2%vsplain3/3negative;allmodalities-8.9%vsplain3/3negative.
  Valmeans4stages.05704/.05602/.05598/.05412;not a stableval/testgain.
  NewRQsemantictriples:headed1025–1088,poster888–953,plot825–889,
  people652–756;plain1271–1273. Firstbookheaded14–15/full10–14vsplain98–140.
  All1273fullSIDs remainunique via suffix. Diversitystillgrows500–3000;
  textMSE~.81–.84vsconstant1,worse~.87–.88withallheads. Do notdeclarecausal
  bookoccupancy rule or modalitiesuseless:weakheadedcontrol/trainingconfounds.
  Plotprep1272descriptions/1285chunks/max657tokens/no truncation,21.3sec.
  See docs/gr-movielens100k-multimodal-results.md and localartifacts/
  movielens100k-multimodal/mpnet-replacement/analysis (reports/plots/ZIP/ranks).
  No pending GPUwork,noautoextensions,tuning,extraseeds,TPUorTelegram.



- MPNet replacement submitted ONCE,v2RUNNING,2026-11-01. Sameprivatekernel
  carasik/rhpr-ml100k-multimodal,oldv1CANCEL_ACKNOWLEDGED. Newexecutable
  addd3c326dbc55ddf18d2dad2c832bf43000a8e0,Actions41313214797successsubmit-only.
  Actualconfig+all186embeddedfiles matchcommit,packageSHA
  c18600490f3f8b805312a891e4061eeabfee3672ac93e154a6fa25b0a0bfc596.
  Onlyplotencoderchanges toMPNet description-only;headedtextcontrol remains
  priorSentenceT5vectors. Frozeninputv2manifest75a4eb4e…verified. Receipt:
  docs/gr-movielens100k-multimodal-mpnet-submission.json. Localartifacts
  artifacts/movielens100k-multimodal/mpnet-replacement. Waitforusercompletion,
  collectexistingv2;noresubmission/continuouspoll/tuning/extraseeds/TPU/Telegram.



- User corrected plot channel,2026-11-01: use MPNET on DESCRIPTION ONLY.
  CachedZenodoall-mpnetvectors mixdescription+genres+tags+reviews perauthors'
  script;notused. Sourcecounts1273genres/1253tags/776reviews. Base text STILL
  exactstage5SentenceT5;onlyplotencoderchanges. Pinnedall-mpnet-base-v2
  e8c3b32edf5434bc2275fc9bab85f82640a19130,max384,chunk/tokenweightedpool+L2,
  no truncation,1272plots/1masked. Real7000token/20chunkMPNet smokeOK;
  native4testsOK,full924OK82skips. Same12fits42/101/202×4stages/20epochs.
  Originalmultimodalv1(source7986c0c)CANCEL_ACKNOWLEDGED,nonecompleted,
  firsttwofitslastloggedepoch6. Preserveoldreceipts;do notresumev1.
  Cancellationoriginnotdetermined;serverstatusverified,no successfulcancel
  APIcallbyagent. Replacement permitted by user's explicit encoder choice.
  Privateinputcarasik/rhpr-ml100k-modal-input v2 READY,MPNetassetsincluded,
  manifest75a4eb4e37ac6909d6c2f68ffe1582bfdc5bde385a6c975adc491ae0c231c0fd.
  See corrected multimodal doc+cancelled-v1.json. Local replacement artifacts
  artifacts/movielens100k-multimodal/mpnet-replacement. One replacement session,
  noextraseeds/tuning/Telegram/TPU. Waitforcompletion aftersubmission.



- Multimodal ML100K session submitted ONCE,v1RUNNING,2026-11-01.
  Privatecarasik/rhpr-ml100k-multimodal,12fits/3seeds×4cumulative channels,
  2T4/offline/noTelegram. Executable7986c0cf1db85610f0878fbe32116a9b311e0b34;
  Actions41310913553successsubmit-only. Actualconfig+all186embeddedfiles match
  committedsource,packageSHA0c26fcaacbd14942dcccc60ced38e1804d6b6e081d7f1eda2bee11ed7ca53980.
  Receipt docs/gr-movielens100k-multimodal-submission.json,localartifacts/
  movielens100k-multimodal. No outputs/quality results collected yet.
  Do not resubmit or continuouslypoll;waitforusercompletion/status,collect
  existingrun. No automatictraining retries/tuning/extraseeds/TPU/Telegram.



- User authorized cumulative multimodal movie experiment,2026-11-01:
  text(nopeople) -> +poster -> +plot -> +people,PLUM-like permodality MLP
  encoder/decoder heads and SUM reconstruction MSE. Config
  movielens100k_multimodal_2t4.json;see docs/gr-movielens100k-multimodal.md.
  Twelve fresh fits42/101/202×4stages,20epochs,val5/10/15/20,previousTIGER4+4
  andsame1273films/943users/split/cases/history20. Textonlyheaded controlfresh.
  SharedRQVAE3x256/3000steps/latent64,all4heads instantiated uniformly;inactive
  blocks zero/loss omitted. Fixed catalogue centering/scalarRMS per channel,
  observed-item dimension-meanMSE summed,noUW. Log perhead/Q losses+occupancy.
  Frozen text=exactpriorstage5T5(noactor/director names);posterViTCLS1273;
  plot=rawdescriptionONLY->pinnedT5,full2048-tokenchunks/tokenweightedpool,
  1272observed/1masked,no cataloguefilter. Frozenpeople=shared64DDeepWalk
  over16692TMDBpersonIDs,same1273films,normalizedfilmwalks,cast/director
  separate means->128D. Gensim4.4/seed42/oneCPUworker,not userinteractions.
  Privateinput carasik/rhpr-ml100k-modal-input READY,manifest
  5894994424f21c54784e416136d77758b5462d370e2a40016633e08eae141bb4.
  Newprivatekernel target carasik/rhpr-ml100k-multimodal,2T4,~2–2.5h+queue.
  Tests924OK82skips,native4OKinclRQ+20epoch tinyTIGER;realT5chunk smokeOK.
  NoextraGPUjobs/tuning/retries/Telegram/TPU. Preserve old evidence/pausedTPU.



- ML100K extra seeds collected,2026-11-01. All14fits completed20epochs;
  rawKaggleERROR is ONLY final aggregation comparing initial NNhash globally
  across101/202. No rootmetrics/summary/runmanifest were emitted. Preserve raw
  failed-run evidence;explicit local recovered-metrics.json is verified,not a
  rewritten successful run. No GPUretry needed/authorized. Config now sets
  compare_initialization_by_seed=true;validator requires it and regression test
  accepts different seeds/rejects withinseed mismatches. Tests920OK81skips.
  Across42/101/202,mean testNDCG10 title.04522±.00173,genres.04149±.00512,
  year.04093±.00616,runtime.04585±.00528,keywords.04143±.00335,
  cast.03839±.00215,directors.04003±.00216. Fulltextloses title3/3(-11.5%).
  Addingcast loses2/3(mean-7.3%);addingdirectors helps2/3(mean+4.3%);
  combinedpeople versuskeywords-3.4%,mixedsigns. Fulltextmeanval.05989>
  title.05613,oppositetestordering. No causalrare-nameencoding failure proven.
  All105savedpredictions rescored;943cases same;everyfit1709400examples/
  6680updates/zeroAMPskips. FrozenT5embeddings bitwiseequal across seeds/stage.
  See docs/gr-movielens100k-seeds-results.md;local evidence/plots/ranks/ZIP in
  artifacts/movielens100k-text-ladder-seeds/analysis. No newGPUruns;graphpeople
  embeddings discussed/audited only,not trained or dispatched.



- Extra ML100K seed session submitted v1 and RUNNING,2026-11-01.
  Private`carasik/rhpr-ml100k-text-ladder-seeds`,14fits/101+202,no repeat42.
  Executable60612f5968131745c23b88b8443bca028a93fdf9,successfulActions41293053201;
  all182embeddedfiles+config independently matchcommit,packageSHA
  d9f16d46a386fe1eb5703ee5da3dda9de22660542f2de2070b6d51704ac6e3f5.
  FirstActions41292795661failed import before submission(missing six in
  KaggleCLI2.2.5);workflow now installs six and smoke-checksCLI. OnlyONE
  actual Kaggle version/run created;no GPUretry. Tests919OK81skips,native4OK.
  Receipt`docs/gr-movielens100k-seeds-submission.json`;localevidence
  `artifacts/movielens100k-text-ladder-seeds`. Do not resubmit or continuously
  poll;waitforusercompletion/status andcollect thisexistingrun. NoTelegram,
  TPU,tuning,extra seeds,automatictraining retries,orolder-studyrevival.



- User authorized TWO EXTRA pipeline seeds per ML100K text stage,2026-11-01.
  Config`movielens100k_text_ladder_seeds_2t4.json`;see
  `docs/gr-movielens100k-seeds.md`. Fourteen fresh fits101/202×7stages,
  one new private`carasik/rhpr-ml100k-text-ladder-seeds`2T4session,seven fits/card.
  Same fixed data/split/case digests/history20/seed-matched20epochs/val5/10/15/20,
  frozenSentenceT5/max2048,RQ-VAE3x256/3000,TIGER4+4/d128/FP16,selectedtest.
  Pipeline seed varies RQ,NN/dropout,epochorder together;withinseed allstages
  matched. Do not rerun42 or tune;no extra seeds/retries/Telegram/TPU.
  Original100KrunCOMPLETE:all7×20epochs,zeroAMPskips,35predictionfiles rescored.
  Title testNDCG10.0471667;keywords bestval.0690003/test.0434149;
  fulltexttest.0394417. One realization,no established richtext benefit/harm.
  Local reports/evidence:`artifacts/movielens100k-text-ladder/analysis`.
  This authorization supersedes the old no-extra-seeds note for this job only.

- Active direction switched to MovieLens100K by user,2026-10-31. ML1M run was
  stopped by user;serverCANCEL_ACKNOWLEDGED,do not resume/relaunch it.
  Newconfig`movielens100k_text_ladder_2t4.json`,same7fields/year-in-title,
  Sentence-T5max2048,RQ-VAE3x256/3000steps/initbuffer1273,previousTIGER4+4/d128,
  20epochs,val5/10/15/20,sameGPU2T4/seed42/history20/strict-past/group-holdouts.
  OfficialML100K100000ratings/943users/1682rawIDs. IDs differ fromML1M.
  Exact title/year aliases and explicit IMDb IDs bridge to previous verified
  metadata;1667matched/13unmatched/2ambiguousrawIDs. No fuzzy/raw-IDjoin.
  CommonTMDBrich4+ViTcoverage1288rawIDs,91767events;15same-film aliases merged
  to1273movies,298duplicateuser-filmratings removed(latesttime,sourceIDtie).
  Final91469ratings,943users,noexcludedusers,2042heldoutgroup embargo events,
  87541train events,85470strict-past train targets/epoch. Dataset preparation
  `scripts/prepare_movielens100k_text_ladder.py`;samecase/modelinit preflight
  passes,NNhash5f4a24fa…unchanged. All7models fresh;noML1Mweights reused.
  Privateinput`carasik/rhpr-ml100k-text-ladder-input`,manifest
  52fa57696f77bc0d137b1d042684afb83e4d17c068fe78c9c5502d573cf84028.
  Newkernel target`carasik/rhpr-ml100k-text-ladder`,not the stopped ML1Mslug.
  About1.5–2hwall estimate,vsold600sec/892574targetsper epoch;quota29.27hleft.
  Tests917passed(81skips),plus3targeted native/config tests. InputREADY.
  Submittedonce:private`carasik/rhpr-ml100k-text-ladder`v1RUNNING,
  executable6537df769c7755a601c7e91672039701c519685d,Actions41235282704success
  submit-only. Actualconfig+all182embeddedfiles matchcommit,packageSHA
  680290105a27db0e88b7e22e82faa006092794f26f9d1940a4725e354fdf8b61.
  Receipts:/workspace/artifacts/movielens100k-text-ladder. Do not resubmit
  or continuouslypoll;waitforusercompletion/status andcollectexistingrun. NoTelegram,TPU,tuning,
  retries,extra seeds,oroldexperiment revival. Keep stopped1Moutputs intact.


- Active authorized MovieLens text ladder,2026-10-31. See
  `docs/gr-movielens-text-ladder.md`,config`movielens_text_ladder_2t4.json`.
  Seven cumulative fields:original title INCLUDING year,genres,explicit Year,
  runtime,TMDBkeywords,all cast,all directors. Stage3 repeats year information,
  per user correction;no overview/user tags/reviews/poster inputs. Frozen pinned
  Sentence-T5,max2048allvariants;actual max1587,no truncation. RQ-VAE3x256,
  3000steps,FP32,initbuffer2875(full smaller catalogue);previous TIGER4+4/d128,
  4977920params,FP16,20epochs,val5/10/15/20,one selected-checkpoint test.
  User approved distinct timestamp-group val/test:one SHA-selected target/group,
  embargo other group ratings11004,exclude user5258(two groups,15events).
  Common2875movies/6039users/906679train events/892574strict-past targets per
  epoch. Simultaneous TRAIN targets share earlier history;no same-time peeking.
  All ratings1..5,history20,fullcatalogue/freebeam20/no seenmask as prior control.
  Private input`carasik/rhpr-ml1m-text-ladder-input`created andready;manifest
  53178c2446d359d00aa864675042cb82fdb98ced421b04a5d86b59046e3bc204.
  Case train/val/test hashes and source identities verified;native initialNNhash
  5f4a24fa…matches previous TIGER. Native real-encoder/RQ/tiny20epoch smoke passed.
  New builder is optional trainer argument;historical defaults remain intact.
  Historical stopped run:private`carasik/rhpr-ml1m-text-ladder`v1CANCEL_ACKNOWLEDGED,2T4
  independent4/3fit workers. Executablec4d31fb32146376fb65f4e043787d2ecbb62c609.
  Actions41230771384successsubmit-only;PR41attached. Actual182embeddedfiles and
  config matchcommittedsource;packageSHA6f7e96b2ee38787d4961fb29139bb7a57c8612598ce80165d13cc77f8c2626f4.
  Receipts/sourceaudit/split/config/tests saved in/workspace/artifacts/movielens-text-ladder.
  Quota30GPUh,estimated~19GPUh/~10–11hwall before launch. Publication/dispatch
  complete;912fulltests passed(12skips),plus2native movie-suite tests.
  Initial PR CI missingSciPy/safetensors from historicalSIDtests;addsid-analysis CIextra,
  no change to already submitted Kaggle executable and no resubmission.
  Prelaunch input statusREADY;kernel searchNotFound. Do not submit duplicates.
  No automatic tuning,retries,extra
  seeds,earlier experiments,TPU or Telegram. Preserve paused TPU edits.


- New direction2026-10-29:rich movie semantics forGR vscontent-aware dense.
  Branch`codex/movielens-rich-semantics`frombd4ba5752d79ef480bf572fdeefbdc129e4dea08.
  See`docs/gr-movielens-rich-semantics.md`,script`audit_movielens_semantics.py`.
  CPU/publicdata audit only,noGPUtraining/embeddinggeneration/Kagglesubmission.
  ML1M6040users/1000209ratings/3883catalogue/3706ratedmovies. TMDBsnapshot
  confirmedIDs metadata3658rated;overview3648,keywords3256,cast3617,director3644,
  posterPATH3649(all/3706). Complete4richtextfields3237. Median5keywords,14cast,
  52overviewwords;not100IMDbtags.10conflictingIMDb/TMDBjoinswithheld,214title/year
  reviewflags,3badmetadatarows,29dupIDs. No fuzzyjoins orinteractionfiltering.
  Posterprobe20fixedseedonly1image/19HTTP404;needcachedmultimodalfeatures orfresh
  images,notoldURLs. Zenodo15403972claims3667plots/3197posters;ML1M-onlyNPY91MB,
  plots24MB,mapping0.7MB,APIinventoryretrieved,contentsnotyetverified. IMDbpublic
  TSVschemaNOplots/keywords/posters. Rawdata/tmp/rhpr-movie-audit,aggregates/
  gzipcatalogueaudit/sourcehashes/workspace/artifacts/movielens-semantics.
  Importantprior:LIGER2411.18814sameiteminformationdensebeatsimplementedGRincl
  cold;MMGRec2404.16555alreadymultimodalGR(notourcontrolledtask). Donotclaim
  literatureempty orGRcoldautomaticallybetter;originalTIGERcoldusesprefixexpansion
  andepsiloncap,currentexactSIDadapterdoesnotimplementit.
  Proposed5fits:SASRecIDcontrol,SASReccontentbasic/rich,TIGERbasic/rich;shared
  encoder/features,comparewithin-familyrich-basic;densecontentbothinput/output.
  Imageslaterafteralignmentaudit. Coldrequiresseparateprotocol,rare!=cold.
  MLratingtimeordercaveat:77.3866%eventsintieduser/secondgroups,53.2149%adjacent
  ties,last2samesec2544users. Holdtimestampgroupstogether;notwatchingorder.
  Oneuseronly2timegroups;3-waygroupsplitneedsdecision. Taskclarificationpending:
  personalizednext-itemvsactualitem2item(userworkmotivateslatter,modelsformer).
  Do not launchtraining before sharedprotocol/caseadapter is concrete.

- Completed teacher-age screen2026-10-28,no pending GPU work. See
  `docs/gr-sid-generator-age-results.md`;private`carasik/rhpr-generator-age-screen`
  v1 COMPLETE,source316c1093a0a76ccf408562c5bce0a8640d72f322.
  Same115pairs(95LCP0/20LCP1/noLCP2),1987754rows/age inclstatic+greedy.
  Static andgreedy accepted2/2/0at1/5/20;changeditems4/4/0. Firsttokenchanges
  2/4/0,secondtokenchanges2/0/0. Gains32.5924/7.47208/0nats,
  relativeTRAINNLLimprovement.002348%/.000682%/0. Affects639/138TRAINcases,
  203/45validationhistory-or-targetusers. Samepair10655/11819serumsstaticgain
  +29.51195/+1.82229/-23.01673at1/5/20;3023/3413nailart+3.0804/-116.174/
  -167.732;8476/10004DevaCurlgel/shampoo-2.116/+5.650/-17.217.
  Ageeffectvisiblebutconsistentwitheitheroldcodeanchoringorearlymodelerrors;
  no downstreambenefitshown,only4changeditemsagain. Do not auto launchfits.
  Oldtwoaccepted20epochpairsarein2000poolbutNOTshared115prefix:zero20here
  doesnotcontradictprior2accepted,differentpool/order/budgetaccounting.
  Only5newtrainingepochs657065cases/2570updates/zeroAMPskips,onevalat5,no test,
  downstreamfits,semanticpenalty,extraseeds,retries. Body30.24min.
  All3checkpointfileweights/epochs/updates/hashesindependentlychecked;3mining
  listsrecomputedfromprobes,pool/budget/order/cachegain/zerooutsideaffectedrows/
  mapreplay/prefixoccupanciesverified. Runtimefrozenweights/RNG/fullcacheaudits
  exactzero. Actual180embeddedfiles/config matchsource;privateofflineT4mounts
  verified,previouscloudoutageauditresolved.PackageSHA04448e74e87ff6bc41485d00d229c65fa121d55aacc089d73f42c9152c942508.
  60numericalfiles11.23MBplus2earlyweightsin/tmp/rhpr-generator-age-completed;
  reports/graphs/paireddeltas/sourceauditin/workspace/artifacts/sid-generator-age.
  Script`scripts/analyze_sid_generator_age.py`reproducesCPUverification.

- Latest authorization2026-10-28:teacher-age1/5/20search-only screen.
  See `docs/gr-sid-generator-age.md`,config`amazon_beauty_sid_generator_age_2t4`.
  Fit original generator exactly5epochs once,save1/5,reuse pinned previous20.
  Mine per-age same16384TRAINrows;balanced roundrobin union2000pairs,shuffle2759.
  Common prefix reserves2xcomplete64-rowbatchcost <=2Mrows/teacher. Eachpair:
  independent original-map static delta AND actual sequential-search delta.
  Allages complete sameprefix;1200sec guard fails partial comparisons. Noaccepted
  cap,semantic penalty,test,downstreamfits,extra seeds,retries/extensions/Telegram.
  Native four-token FP32likelihood,FP64sum,history+target remap,same thresholds,
  frozenweights/RNG andfullcacheaudits as previousscreen. Staticbranch doesnot
  combine moves;greedytrajectories maydiffer andmustnotbe confused withstatic.
  ExactsameinitialNNhash/seed42/data/order/native5epochtraining;2T4workers
  searchages1/5on0and20on1. Quota7.97hbeforelaunch,expected30–45min+queue.
  Source/indices allpinned,noexistingjobresubmission. Tests905OK80skips;4targeted
  passincludingnative5epochsnapshots/staticprobes/3isolatedsearches;nativefirst5
  lossesmatchwithoutsavehook. Sharedprefixdrycheckwith20-onlypool119pairs,
  1998506reservedrows(drycheckonly). Submitted v1,now complete;results above.
  Do not proceed to downstreamtraining automatically.

- Completed generator-guided SID screen,2026-10-28. No pending GPU work.
  See `docs/gr-sid-generator-guided-results.md`; executable c3459221dfae7163b46de0d9ec28c4003a679735,
  private `carasik/rhpr-generator-guided-sid`v1 COMPLETE. Search214proposals,
  2accepted/4changeditems(0.0331%),1990375scoredrows;stopped rowbudget,535.6sec.
  All209LCP0/1proposals rejected with negative gain;both accepted share first2
  tokens and only change third. Pairs328/2126brushes,4943/5058Proactivsizevariants.
  Frozen TRAIN NLL7.452504137->7.451465099(-0.01394%),gain136.543167nats,
  exactzero full-cache/identity drift,weights/RNG unchanged. Affects873TRAIN
  cases,193valusers(157history/36target). Both freshfits sameinit/seed42,
  20epochs/2628260cases/10280updates/zeroAMPskips/4977920params/no test.
  Original/guidedNDCG10at5=.0256211/.0268026(+4.61%),10=.0294257/.0297161
  (+.99%),15=.0341011/.0334771(-1.83%),20=.0364335/.0353168(-3.07%).
  Endpoint delta=-.0011167,paireduser95%CI[-.0023401,+.0000804],targetcluster
  [-.0029946,+.0007088],2000resamplesseed2760,conditional fittedweights/no seedCI.
  ValteacherNLL8.792829->8.787134slightlyimproves despite retrievaldrop.
  98.3%netdropamongunchangedvisibleinputs/targets:sharedtraining,notcausalproof.
  No benefit established,also no stable harm;tiny withinbranch intervention
  does not reject generator-guided upper-tree learning. Frozen-teacher anchoring
  is a plausible limitation,not proven cause;do not auto extend search or launch.
  68numericalfiles32.10MBin/tmp/rhpr-generator-guided-completed;checkpointsstay
  privateKaggle. Analysis/plots/peruserranks/search/sourceauditsin
  /workspace/artifacts/sid-generator-guided;8beams+8TFarrays independentlychecked,
  replay/owners/prefixoccupancies/cache/trainstreamverified. Body41.0min.
  Script `scripts/analyze_sid_generator_guided.py` reproduces CPUanalysis.

- Latest authorization,2026-10-28:one frozen-generator SIDsearch then twofresh
  original/updated20epoch fits,val5/10/15/20,on2T4. See
  `docs/gr-sid-generator-guided.md`,config`amazon_beauty_sid_generator_guided_2t4`.
  Sourcecorruptionkernelrandom_p000best20weightsSHAdeadfd084b77f4a68a0f39097a423d187316b94a9b2a4d1ea563639efbf303c9,
  19977481bytes,epoch20/10280updates;parentmanifest/source/table/resultpinned.
  Exactfour-tokenFP32teacherforcinglikelihood,microTRAINsum,historyANDtarget
  remapped. Fixed64-rowbatchgroupscache,alltarget/historyaffectedrows,no beam
  scoreorvalid-childrenormalizationinacceptance. Positivegain>.01natsand1e-5
  peraffectedrow;fullTRAINcacheaudit<=1e-5perrow/.02sum,weights/RNGfixed.
  Proposalmining:16384TRAINrowsseed2758,freegreedy1path,validwrongitems,
  teacher-forcedlogmargin>=log1.25,min2TRAINtargets/item,clippedgappriority.
  Fixedlist,max2000proposals/256accepted/2Mscoredrows/1200searchseconds,stopfirst
  cap,noneforcing/no retries. Zeroacceptsskipbothfreshfits. No semanticpenalty,
  sourceencoder/quantizerrefit,test,extraseeds,extensionsorTelegram. Fresh2fits
  exactprevious4+4/d128FP16protocol,sharedinitialhash5f4a24fa…,seed/order42.
  Saveall8beams,best/laststates,TFsidecars,searchtrace,candidates,TRAINcache/maps.
  RealcheckpointCPUpreflightidentity/loss/frozenstate/RNGpass;901testsOK79skips,
  fourtargetedOKincludingrealparentsearch+2freshworkers+zerochangebranch.
  Submitted once:private`carasik/rhpr-generator-guided-sid`v1,now COMPLETE.
  Actions40663408472success,executablec3459221dfae7163b46de0d9ec28c4003a679735.
  Actualsource/config/all179packagefilesverified;packageSHA256
  d64fd323278a1b422b1bd07ba403b0e10880578a2ad99216d7a3a75731fd19f4.
  Receipts/preflight:`/workspace/artifacts/sid-generator-guided`.
  Do not submit another version or poll continuously;completed analysis above.
  Quota8.66hbeforelaunch. Expected45–65min+queue. Researchquestion
  isindextransfer,notJTMreproductionorteacherNLLguaranteeingNDCG.

- Completed ALSfootprint audit,2026-10-28,CPUonly:no newfits/inference/test/GPU.
  See`docs/gr-sid-als-footprint.md`,artifacts`/workspace/artifacts/sid-als-footprint`.
  Mainuniform480items:fullTRAINhistusers4441/22363(19.86%);visiblevalhist4306
  (19.26%),target757(3.39%),either4927(22.03%);trainprefixeither28268/131413
  (21.51%). FinalLCP0/1/2/3=401/77/2/0:83.54%changefirsttoken.256movesreplayexact.
  FrozenT5destination-ownercosinemedian.87917vsrandomsame480.85161,catalogue
  .84860;mediansemanticrank1304.5/12100,33top20/76top100. Cosineisitemvsold
  ownerofnewpath,notchangedT5vectororcentroid.100untrainedsame480/pathsetrandom
  controls:rootchangedmean475.49vsALS401;semanticLCP.199945original->.196909ALS
  vs.192671random. FitALSgraph.045589->.047448vsrandom.044082;selectedbyconstruction,
  notqualityproof. MovedmedianTRAINtargets5vs6,mean9.43vs10.92;mass3.45%.
  Popularitybins0/1-4/5-9/10-19/20-49/50+=2/190/180/63/35/10.
  Other4%variantsverydifferentexposure:priority472itemsvalunion34.68%,context20
  48231.51%,context548132.33%;random5%60528.55%. Prioritymedianfreq12,mass7.61%.
  Affectedmain4927qualityfresh20-8.04%,longrun20-.97%,longrun60+5.97%(CIs0).
  Existingpriority10vsrandomsame472affected7756+8.81%,userCIpositivebuttargetCI0;
  global+.94%CI0. Exploratorylead,noconfirmedusefulCFgain. Sharedweightsaffect
  unchangedinput/labelrows,notcausalexposuredecomposition. Artefactssaveall480
  itemfeatures/distances/paths,pair256steps,22363exposureflags,100randomdiagnostics,
  PNG/SVG. No trainingauthorized;waitfornextresearchdirection.

- Completed randomSIDcorruption,2026-10-28:no pendingGPUjobs. See
  `docs/gr-sid-corruption-results.md`,artifacts`/workspace/artifacts/sid-corruption/analysis`.
  All5fits20epochs/2628260examples/10280updates/zeroAMPskips/4977920params,
  sameinit/data/order42,permutation2753,no test. NDCG10at0/5/20/50/100%=
  .03644569/.03362916/.02924565/.02174110/.01513320;relative-7.73/-19.76/
  -40.35/-58.48%. Doseorderingall4checks,notindependentreplicates. Allpaireduser
  andtargetcluster95%CIsnegative,conditionalfixedweights/unadjusted,notseedvariance.
  TeacherNLL8.77067->9.43382;first-tokenNLL4.46815->5.17095,top111.61->1.78%.
  LaterlevelsnetNLLimproveundergiventrueprefixes;notmerelybeampruning,beamwidth
  effectstillpossible. Invalidbeams333->0of447260. Catalogueoccupanciesall4depths
  exactlyfixed;TRAINprefixfreqNOTfixed,firstTV.18072,entropy7.44586->7.47854bits.
  T5first-prefixneighbors53.22->0.62%. Originalorganizationhelpsat20epochbudget,
  notpuresemanticcausalityorasymptoticproof. Allfitsstillgrow;100%15->20epochs
  .012471->.015133. ALSswapbenefitnotestablishedbythisrandomizationresult.
  All20beamsand20teacherarraysindependentlychecked;source/config/stream/neural/
  map/ownerhashes,nestedderangements,semantic/frequencydiagnosticsverified.
  Raw135numerical/reportfiles70.17MBin/tmp/rhpr-sid-corruption-completed;
  authoritativeKaggleretains10statefilenames,contentsnotdownloaded/verified.
  Body93.2min. Source1e87396a…. ScriptsreproduceCPUanalysis,PNG/SVG+all20ranks
  retained. No newtraining/inference/test/Telegram;waitfornextresearchdirection.

- Submitted randomSIDcorruption once,2026-10-28:private
  `carasik/rhpr-sid-corruption-screen`version1RUNNING. GitHubActions40595947524
  succeededsubmit-only,noTelegram. Executable1e87396aa65fd8b2dc9259a650e501cec25eec35.
  ActualKagglesourceASTexactconfig+all178embeddedcommittedGitblobhashesverified,
  package3f4491646dc4bdd907de55f76f7f6d53832144e8256d8ebfcd9f0a80aafcc542.
  Private/GPU/T4/offline/3existingdatasets/TPUoffverified. CPUpreflightall5neural
  initialhashesmatchandSID/ownerhashespinned;897testsOK(78skips),4targetedOK
  includingrealassets+nativefits+teacher/RNGguards. Receiptsunder
  `/workspace/artifacts/sid-corruption/`(preflight,submission,source-audit).
  No resultyet. Five20epochfits,val5/10/15/20asbelow. No resubmission,retry,
  extension,extraseeds,Telegramorcontinuouspolling;waitforusercompletion/status.

- Latest authorization,2026-10-28:random whole-SID corruption0/5/20/50/100%,
  five freshTIGERfits20epochs,val5/10/15/20,on2T4. See`docs/gr-sid-corruption.md`
  and`amazon_beauty_sid_corruption_2t4.json`. SameoriginalRQKM,neural/order42,
  permutation2753,nestedselectedsets;uniformderangementswithinselectedcounts
  0/605/2420/6050/12101. Completepathsincl suffix. PerdoseSID/ownerhashespinned
  fromCPUpreflight;all5neuralhashesmatch5f4a24fa…,4977920params,131413cases/epoch.
  Everycatalogueprefixoccupancyexactlyfixed;TRAINtokenfrequencyNOTcontrolled.
  Existing3privatedatasets,originalinput+T5semanticgraphmanifest/cataloguechecks,
  no uploads/refits. T5first-prefixsharing53.22/48.06/33.98/13.68/0.62%,TRAIN
  firstprefixTV0/.01690/.05815/.10519/.18072. No downstreamresultyet. Save20beams,
  20teacher-forcedNLL/ranksidecars,10best/laststates,allmaps/owners/frequencies/
  diagnostics. Teacherpassobservational,fulltrue-prefixheads,notbeamquality.
  Jobs[0,20,100]and[5,50],expected85–110min+queue. No test,earlystop,extension,
  extraseeds,retries or Telegram. Launchpending;quota10.23hbeforelaunch.

- Completed identical-index seed control,2026-10-28:no pendingGPUjobs. See
  `docs/gr-sid-seed-oracle-results.md`,artifacts`/workspace/artifacts/sid-seed-oracle/analysis`.
  SameoriginalRQKM,neural42/43,data/order42,both20epochs/2628260examples/
  10280updates/zeroAMPskips/4977920params/no test. All8rawbeams independently
  rescored,NDCG/Recall5/10,invalid/duplicatecounts,cachedranks/oracle match.
  Seed42NDCG10=0.03644111,seed43=0.03621833,oracle0.04785565(+31.32%).
  Oldoriginal/ALS20oracle0.04729654(+28.03%). Same-indexexclusivehits404/382,
  shared1196,miss20381;oracleRecall10=8.8629%(1982hits),oldpair1984hits.
  Same-indexheadroomat5/10/15/20:+29.55/+32.55/+30.30/+31.32%;notindependent
  replications. LargeoracleoccurswithoutSIDchange;noadditionalhierarchy-specific
  oraclebenefitshown. Doesnotquantifycausalshare,proveequivalenceorlearnable
  routing. Headroomdifference(same-different)+0.0010606,user95%[-0.0000441,
  +0.0021789],targetCIalso0;fixed-weightintervalsnotseedvariability. Fresh42
  differsfromold42despitehash/protocolmatch;GPUtrainingnondeterminismremains.
  Raw46smallfiles25.88MBin/tmp/rhpr-seed-oracle-completed;fourstatefilenames
  retainedonprivateKaggle,contentsnotdownloaded/verified. ScriptsreproduceCPU
  analysis;no newtraining/inference/test/Telegram. Waitfornextresearchdirection.

- Submitted identical-index seed oracle once,2026-10-28:private Kaggle
  `carasik/rhpr-sid-seed-oracle`version1RUNNING. GitHubActions40589179704
  succeededsubmit-only,Telegramskipped. Executable178883fe87351ee81ea70830688782129370584d.
  ActualnotebooksourceASTverifiedexactconfig+all177embeddedcommittedGitblob
  hashes,package321c269addc30abe861ca4b3a8197e0e2031872d9d928b5c33279b961b80b4b0.
  SameoriginalSIDfileb196711dceedce81a991d995aa5c86ccd35288d5cdfe3f7aedb174de86470dae.
  Seeds42/43neuralonly,shareddata/order42. Each4977920params20epochs,val5/10/15/20,
  2T4offline/twoexistingprivatedatasets/no test. Oraclecomputedfromsavedbeams
  atcompletion,not a learnedrouter. Standard893testsOK(76skips),3targetedOK
  includingrealparent+isolatedCPUworkers+checkpointedfits+oracle,mispairguards.
  Receipts`/workspace/artifacts/sid-seed-oracle/{submission,preflight,source-audit}.json`.
  Quota10.73GPUhoursbeforelaunch. No resultyet;do notresubmit/retry/extend or
  continuouslypoll. Waitforusercompletion/statusrequest. PausedTPUfilesuntouched.

- Latest authorization,2026-10-28:identical originalRQKMeans index,two fresh
  TIGERgenerator seeds42/43,20epochs,val5/10/15/20,on2T4. See
  `docs/gr-sid-seed-oracle.md` and `amazon_beauty_sid_seed_oracle_2t4.json`.
  Fixed data/order seed42 and exact old frozen original SIDbytes. No quantizer,
  ALS,swap,branch token or test. Preserve all8validation beams and full best/last
  states. Independently score best-of-two NDCG10 after fits,save paired-oracle
  and per-user ranks. One pair controls ordinary neural-training diversity,
  not a causal index-gain estimate. No extension,retry,extra seeds or Telegram.
  Shared worker launcher groups initialization checks by generator seed;worker
  expected hashes pinned per trial,TRAINstream remains common. Launch pending.

- Completed gatedindex analysis,2026-10-27:no pendingGPUjobs. See
  `docs/gr-sid-gated-index-results.md`,artifacts`/workspace/artifacts/sid-gated-index/analysis`.
  Both20epochs/10280updates/zeroAMPskips/4978176params. All8rawpredictionfiles
  independentlybranch-decoded,allmetrics/invalid/duplicatecounts/user-target-
  branchmetadata/hashesmatched. ControltaggedoriginalNDCG10 0.03587286,gated
  0.03603699(+0.46%);user95%deltaCI[-0.001110,+0.001453],targetclusteralso0.
  Overallrelativeat5/10/15/20:+2.35/+4.32/-2.30/+0.46%. Endpoint<10+0.10%,
  10+ +2.05%;finer3+4.06%,4+0.86%,5–9-3.96%,10–19+5.92%,20+-5.98%.
  ALLcohortuser/targetclusterCIsinclude0. Current20+control0.046972,gated
  0.044165;olduntaggedoriginal/pureALS0.039276/0.044327arehistoricalrefs,
  notmatchedfeatureablation. Only116longvalidationtargets(74items)changeSID,
  control7vs gated4top10hits;mostmapassignmentsagree96%. Longchangedinput/
  targetunion1538hasNDCG0.039221->0.037175;unchanged2009 0.043219->0.046288.
  Sharedweightsgloballyaffectunchangedpaths,notproofignoredgate/noise. No
  confirmedqualitygain/equivalence,newthreshold,orindependentseedreplication.
  Source45d7dab1…,45numerical/reportfiles25.98MBcached/tmp/rhpr-gated-index-completed;
  authoritativeprivateKaggleretains4best/laststates,filenamesconfirmednotcontents.
  All8userranks+branch/length/support/pathflagsNPZ,tables/CIs/figure preserved.
  Reproduce`scripts/analyze_sid_gated_index.py --plot`. No newGPU/inference/test/
  training/routingchangeorTelegram;waitfornextuserresearchdirection.

- Submitted gatedindex once,2026-10-27:Kaggle
  `carasik/rhpr-gated-sid-tiger-screen`version1RUNNING. GHworkflowdispatch401
  was notaccepted/no workflowrun;usedofficialKaggleCLI directly,not a duplicate.
  Executablecommit45d7dab1d35a55821585e17423f32eb197a5cce3. All176embedded
  filesmatchedGitblobhashes;unrelatedpausedlocalfile substitutedonlyinmemory,
  worktreepreserved. ActualKagglesourceASTconfirmedexactconfig+packagehash
  c358125a17c2a447d96aa56f142c1718771285861f08ec59ad538fdcc4592f85.
  Private/T4/GPU/internetoff/twoexistingdatasets metadata verified. Receipts
  `/workspace/artifacts/sid-gated-index/source-audit.json`and`submission.json`.
  Exactly2fits20epochs,val5/10/15/20asbelow. No resultsyet;no resubmission,
  retry,Telegram or continuouspolling. Waitforusercompletion/statusrequest.

- Latest authorization,2026-10-27:conditional original/ALSindex,20epochs,val
  5/10/15/20. See`docs/gr-sid-gated-index.md`,config`amazon_beauty_sid_gated_index_2t4`.
  Exactly2freshfits:original_with_branch(originalbothbranches)andgated_original_als
  (originalforavailableH<10,exactolditemALSforH>=10). BothsameknownSHORT/LONG
  learnedencoderprefixtoken,insertedbeforeuserAFTERnativeSIDhierarchyoffsets.
  4978176params(+256),samecase/order/seed42/optimizer/FP16train/FP32batch32
  freebeam20. Sourceinputexistingprivatecontextscreenmanifest8878eb9f…,
  no refit/upload/contextvariants/test/Telegram/retry/thresholdsearch. Gateper
  currentprefix,not eventualuserlength. Per epoch98608short/32805long;shared
  conditionedneuralhashb5700937900c8bfd4f0665ca20937da4727a4b2e7c32346835915338eab81b6d.
  16tests passincludingrealGRIDtrain/gradient/beam/restore,legacyregressionand
  actualcataloguepreflight. Branch-awaredecode+8predfileswithbranchIDs/lengths,
  best/lastfullstateshashBOTHmappings. Future replaymustuseconditionalfactory,
  notoldsingleSIDdecoder. Preflight`/workspace/artifacts/sid-gated-index/preflight.json`.
  Quota11.24hbeforelaunch,oldscreenCOMPLETE;expected35–45minexecution+queue.
  Kernel`carasik/rhpr-gated-sid-tiger-screen`,exactlyone2T4submission.

- Completed contextALSscreen analysis,2026-10-27:no pendingGPUjobs. See
  `docs/gr-sid-context-screen-results.md`,artifacts`/workspace/artifacts/sid-context-screen`.
  All4fits20epochs,10280updates,zeroAMPskips,16validationfiles independently
  verified,sharedinitial/stream/indexhashesmatch. All-userNDCG10 original
  0.036943,itemALS0.035149(-4.86%),context20 0.035796(-3.10%),context5
  0.034211(-7.39%).20+TRAINevents cohort1019users:original0.039276,
  itemALS0.044327(+12.86%),context20 0.045146(+14.95%),context5 0.041013
  (+4.42%).All20+user/targetcluster95%CIsinclude0. Newcontext20vsitemALS
  only+1.85%,alsoCIincludes0. Context20negative5/10/15,positiveonly20;
  context5positiveon20+all4checks,butnotindependentreplication. Otherlength
  binsnegativeat20. Modelseesmin(h,20),differentusercohortsnotcausaltruncation.
  Paired2000bootstrapsseed2740+cohort,posthoc/unadjusted/onefitseed. CPUgraph
  proxiesimprovedwithoutaggregateTIGERgain. Tables/2figures/all16userranks/
  target-support/pathchange flags/PQ+NPZsaved.73raw/smallfiles51MBcachedin
  `/tmp/rhpr-context-screen-completed`;authoritativeprivateKaggleretainsall8
  best/laststates,filenamesconfirmedbutweightsnotdownloaded. Scripts
  `analyze_sid_context_screen.py`/`report_sid_context_screen.py`reproduceCPU
  analysis. No newinference,training,test,routingrule or experimentlaunched.

- Submitted contextALS20screen once,2026-10-27:Kaggle
  `carasik/rhpr-context-als-tiger-screen`version1QUEUED. GitHubActions40541334398
  succeeded submit-only,noTelegram. Executablecommit9dcee38f1fe77cff18daf093246c7c81e9dc0751
  confirmedinactualKagglesource;private/GPU/T4/internetoff/TPUoff/two datasets
  metadata confirmed. Receipt`/workspace/artifacts/sid-context-screen-submission.json`.
  Four20epochfits,val5/10/15/20asbelow. No results yet,do notresubmit while
  queued/running. Nextactiononlywhenuserreportscompletion orasksstatus.

- Latest authorization,2026-10-27:four matched20epoch TIGERfits,validation5/10/
  15/20,one private2T4session. See`docs/gr-sid-context-screen.md` and
  `experiments/amazon_beauty_sid_context_screen_2t4.json`. Fresh originalRQKM,
  exact prioritemALS256swap control,contextALSwindow20and5cosine-neighbor
  candidates. EachGPUrunscontrolthencontext;sharedseed42/exactprevious
  data,initialization,stream,optimizer/FP16train/FP32batch32freebeam20eval.
  Primaryendpoint20,no test,tuning,extra variants,seeds,retries or Telegram.
  Saveall16validationpredictions,best/full-last20states. No intermediate
  5/10/15weights promised. Newprivateinput`carasik/rhpr-sid-context-screen`,
  manifest8878eb9f8b3f5f3174a763e501d8387f1c85b298651e6485b6638b9c87782209.
  Two realasset/launch tests andfive existingALS/nativecadence tests passed.
  Quota12.26hoursbefore submission;prior replayCOMPLETE. Expected60–80min
  execution,queueadditional. Kernel`carasik/rhpr-context-als-tiger-screen`.

- Context-ALS CPU implementation,2026-10-27:user requested typical preceding
  history embeddings per target item and same SID swaps. No GPU launch. See
  `docs/gr-sid-context-signal.md`,config`amazon_beauty_sid_context_prepare_cpu`.
  Frozen ALS raw means within H,then equal context mean per target;TRAIN-only
  same75/25users,primarywindow20/sensitivity5. Minimum2distinct context users,
  reliability u/(u+10),positive cosine top20 BEFORE reliability edge weighting.
  Weighted-cosine neighbor ranking audited separately:top100incoming mass41.46%
  vs corrected11.91%,oldALS4.32%. Both final candidates256swaps,482/481changed.
  Heldout TRAIN count-model gain original0.411252,oldALS0.413672,context20
  0.447136/context5 0.444777bits;not NDCG or validated quality proxy.219items
  no fitcontext,875only1user;34%short1–2contexts. Semantic distortion differs,
  and confidence policy differs from old ALS:do not claim isolated CF effect.
  Artifacts`/workspace/artifacts/sid-context-signal/cosine-neighbors` include
  curves.png,allitem supports,primaryvectors/graph,candidatepaths/owners/moves,
  provenance and support-controlled reliability.17tests pass,1native skipped.
  ALS20control should join next substantive generator experiment,not separate
  quota-consuming reconstruction. No new generator session authorized here.

- ALS20clarification,2026-10-27:long-run epoch20 predictions exist for both
  original and same frozenALSindex;weights best/last are60only. Normalval20
  original0.03642522 vs ALS0.03637116(-0.15%),cooc20pair-3.88%,sameALSpair60
  +2.25%. JointuserCI for ALS-vs-cooc effect at20 and ALS effect change20to60
  both include0. Index budgets256ALSvs62cooc accepted swaps also differ.
  Cannot redo truncated-context inference at ALS20from saved beams alone.
  Cooccode is normalized binary-user cosine;knownproblem sparse edge support
  88.3%with1common user,not proven missing normalization. Read-only audit:
  `/workspace/artifacts/sid-context-robustness/als20-audit/README.md`. No newGPU
  inference/training authorized or launched for this clarification.

- Completed robustness,2026-10-27:original d4305b9c session completed10models/
  40passes,zero fitting/test,no replacement/cancellation. Firstlog34.65sec,
  runtime33.86sec,body1077.88sec,whole1117.78sec;externalwait/stdoutsilence
  unresolved,not evidence of piphang. All40rawbeamfiles verified,allparent
  predictions exact,statesunchanged. See`docs/gr-sid-context-robustness-results.md`
  and `/workspace/artifacts/sid-context-robustness/analysis/README.md`.
  All22363users:20epochRQKMeans/cooc3to20interaction-0.002195userCI
  [-0.003255,-0.001106],targetCIalso negative.20epochRQVAE/coocflat
  -0.000144CIincludes0. Allfour10ALS/random interactionsCIinclude0.
  Thus60ALSpositive findingisconfiguration-specific,notuniversal.20/60indices
  differinsignal/budget,notisolatedduration. Allseed42/sharedcontrols;no
  independentseedclaim.1,019matchedcohort also heterogeneous.894520rowParquet
  andallcurves/CIs/contributions retained. Prepared startup/loggingfixes were
  notusedinthisrun. No pendingGPUjobs or newexperimentauthorization.

- Historical diagnostic before completion,2026-10-27:context robustness source d4305b9c is reported
  Running with empty stdout after>30min. API also RUNNING/no persistedoutputs;
  CLI liveprobe noevents,SDK logstream400no body. Rootcauseunknown. GlobalGPU
  quota increased;user askedtoStopviaUI because cancelAPIrequires sessionID
  unavailablefromstatus/metadata. Do notsubmitduplicate whileold RUNNING.
  Prepared startup-onlyfixes:Python markerbeforewrites/bootstrap,GRprepare/
  pip/importprogress,piptimeout300sec,lazyparentrglobfallback,validationprepare
  phases.29tests pass. See`docs/gr-sid-context-startup-diagnostic.md`. Models,
  cohorts,FP32batch32 and40inferencepasses unchanged. Resumeoriginaltaskonly
  after oldstopconfirmed;no newtraining/tuning or extra experimentalvariants.

- Latest authorization,2026-10-27:context robustness on all10retained models,
  user explicitly chose full15–20min inference scope. See
  `docs/gr-sid-context-robustness.md` and
  `experiments/amazon_beauty_sid_context_robustness_2t4.json`. Four epoch20
  original/cooc-swap models acrossRQKMeans/RQVAE; six epoch10ALSuniform/
  priority/random models. Original10checkpoints do not survive; compare10ALS
  againstsame-family random,not later originals. Full22363-user cap policies
  min(K,h) plus fixed>=20/9/5cohorts,all3/5/9/20passes with samebatchlayout.
  Parent/code/checkpoint/SID/selection/reference hashes and pertrialepoch/budget
  guards. Exactlyone2T4session,40validationpasses,zero fitting/optimizer/test.
  Reuse60data locally:all-user interaction3to20+0.0016153CI[0.000389,0.002875].
  Do not pool different controls or call shared-seed cases independent repeats.
  No extra training,uploads,candidates,seeds,retries or Telegram.

- Completed context ablation,2026-10-27:no pending GPUjobs. Source32b7ce00,
  kernel`carasik/rhpr-context-length-replay`. See
  `docs/gr-sid-context-ablation-results.md` and local artifacts
  `/workspace/artifacts/sid-context-ablation/analysis/README.md`. Eight inference
  passes,zero training/test,227secbody. Primary same1019history>=20users:
  swap NDCG10change at L3/5/9/20=-9.78%/+1.50%/+9.20%/+18.51%.
  Prespecified3to20interaction+0.0140825,userCI[0.003764,0.023628];
  target-block sensitivityCI[0.002382,0.026132]. Secondary history>=9and>=5
  interactions positive butCIoverlap0. Both models peak in absolute quality at9,
  then decline at20. Exact parent replay,unchanged-input controls and weight
  hashes all pass;all8rawbeamfiles independently verified in memory. One fitted
  epoch60pair only,not seed/epochreplication or a pure semantic/CFcomparison.
  Data/Parquet/contributions/figures retained;no next experiment authorized.

- Latest authorization,2026-10-27:inference-only fixed-user context truncation.
  See `docs/gr-sid-context-ablation.md` and
  `experiments/amazon_beauty_sid_context_ablation_2t4.json`. Exactly the two
  frozen epoch60checkpoints from `carasik/rhpr-rqkmeans-als-long`,source10e716df;
  pin source/checkpoints/SIDs/selection/referencepredictions. No fitting,optimizer,
  checkpointselection,indexupdates,testevaluation or upload. One model per T4,
  FP32batch32/freebeam20. Mask old events toPAD in the original20positions,
  retaining identical full-user batch layout across lengths3/5/9/20. Primary
  cohort1019users history>=20usesall4lengths;secondary4242history>=9uses3/5/9
  and10980history>=5uses3/5. Primary interaction is change in swap advantage
  from3to20within the same users/targets. Save allpredictions and compactuser
  metrics,verify modelunchanged and exact outputs on unchanged-input rows;
  report parent replaydrift. No otherGPUjobs,seeds,retries or Telegram.

- Completed long-run analysis,2026-10-27:the two60epoch fits finished;no pending
  launches. Artifacts `/workspace/artifacts/sid-als-long/analysis/README.md`.
  Epoch60validation NDCG10 original0.03985706 vs ALSswap0.04075540(+2.25%),
  paired-user interval for delta[-0.000694,+0.002469];all6checkpoint intervals
  overlap0. Late30/40/50/60mean gain+0.375%.987better/961worse at60;
  descriptive>=3/4late checks with no opposite signs gives281better/288worse,
  1126mixed-sign users. Full user/item Parquet,complete stable-group CSVs,
  extrema,histories,paths and all12validation predictions saved. Both fits
  consumed7884780examples;7AMPskips each,30833actual updates out of30840.
  Earlier rarebrush10906anecdote is not replicated:both hit26/26at60;rare
  TRAINtarget-count1–4cohort has41vs37hits. History5–9has consistent late
  descriptive gain;multiple post-hoc cohorts/one fitted pair prevent a claim.
  TRAIN-only exact-ID context-profile probe also saved:ALSalready improves
  first-branch last5contextcoherence~0.97%,but low-support profiles have poor
  disjoint-user reproducibility. `context-hypothesis.md` formulates a frozen
  prefix encoder/context-distribution graph with confidence shrinkage;it is
  a proposal,not a launched experiment. No new training,swaps,test evaluation,
  GPU sessions or Telegram were used for this analysis.

- Latest authorization,2026-10-27:two longer fits,original RQ-KMeans versus
  the latest screen's ALSuniform candidate. User chose60epochs and then
  explicitly corrected validation to every10epochs. See
  `docs/gr-sid-als-long.md` and `experiments/amazon_beauty_sid_als_long_2t4.json`.
  Two fresh matched fits,one per T4,shared seed42,GRID4+4/d128,FP16,all-prefix
  sampling. Frozen indices reuse the existing private ALSscreen dataset;no
  graph/quantizer/encoder refit or upload. Explicit original-path control keeps
  manifest-pinned original bytes and identity ownership. Primary endpoint is
  epoch60validation,also retain10/20/30/40/50trajectory and best separately.
  Save all per-user validation predictions and full last state. No test access,
  extra candidates,seeds,tuning,automatic retries or Telegram. Earlier screen
  restrictions are historical for its completed run,not a block on these fits.

- Latest authorization,2026-10-27:screen the next graph hypotheses with10TIGER
  epochs,validation5/10. See `docs/gr-sid-als-screen.md` and
  `experiments/amazon_beauty_sid_als_screen_2t4.json`. Exactly six new fits:
  two frozen T5index families x ALSuniform/ALSpriority/random-on-priority-itemset.
  Reuse original/legacycooc epoch10controls. ALSimplicit0.7.3/64factors/.1reg/
  alpha20/20iterations on fixed75%TRAINusers;no encoder/quantizer updates.
  Max100kchecks/cap256accepted;all guided trajectories reached256. Preserve
  exact full paths and suffixes. Shared seed42,4+4/d128,FP16,all-prefix sampling.
  Primary score is epoch10validation,even if best5/10differs. Save per-user
  validation at5/10 and full last checkpoint with optimizer/scaler/allRNG for
  later continuation. No test access/evaluation or automatic20epochfinalists.
  Private derived public-Amazon inputs:`carasik/rhpr-sid-als-screen`. No raw
  reviews,text,names,email,credentials;transfer audit under local artifacts.
  Do not launch otherCFmethods,tuning,seeds,retries or Telegram. Field stays closed.

- Latest request,2026-10-27:inspect user-level validation differences and explain
  graph proposal selection/alternative signals. Training saved val aggregates
  only. Prepare one short inference-only replay on the four frozen epoch20
  checkpoints;see `docs/gr-sid-validation-replay.md` and
  `experiments/amazon_beauty_sid_validation_replay_2t4.json`. CPUfull replay
  costs about3hours;two-T4 replay expected5–8minutes. No optimizer/training,
  model selection or test pass. Pin every checkpoint/SID/report;FP32batch32,
  beam20,preserve complete paths and record any aggregate numerical drift.
  Save per-user val predictions/ranks/metrics for all22363users. No other
  training,CF-method experiments,retries or Telegram. Current graph is sparse:
  88.3%top20edges have one shared TRAIN user;this is a hypothesis,not a cause.

- Latest authorization,2026-10-27:run TIGER20epochs on the completed graph
  candidates atlambda0.3,k20,20kproposals. Use
  `experiments/amazon_beauty_sid_graph_tiger_2t4.json` and
  `docs/gr-sid-graph-tiger.md`. Exactly three new fits in one private two-T4
  session:original/swapped RQ-KMeans on the same worker,swapped native RQ-VAE
  on the other. Reuse completed original RQ-VAE source6e37388a,whose seed/data/
  generator/training configs match;old RQ-KMeans100epoch run is not a control.
  Frozen index dataset:`carasik/rhpr-sid-graph-indices`,manifest3df39c67….
  Preserve exact complete SID paths INCLUDING suffix viafrom_full_sids;no
  centroid assignment,suffix regeneration,encoder/quantizer/graph fitting.
  Shared seed42,4+4/d128,FP16,batch256/physical128,all131413prefix cases,
  20complete epochs,val10/20,NDCG10selection,one FP32freebeam20test. Check
  initial neural5f4a24fa…and case6f7ba06c…hashes. Save full checkpoints,
  curves and per-user test predictions. No otherlambda,k,InfoNCE,extra seeds,
  tuning,automatic retries or Telegram. Field branch remains closed.

- Latest request,2026-10-27:the Field RQ-VAE branch is closed. The active
  experiment is CPU-only fixed-tree SID graph placement;see
  `docs/gr-sid-graph-swap.md` and
  `experiments/amazon_beauty_sid_graph_swap_cpu.json`. Reuse frozen full-metadata
  Sentence-T5 embeddings and saved native RQ-VAE/normalized-residual RQ-KMeans
  indices. Swap complete paths including suffix;preserve all prefix occupancies.
  Fixed semantic/CF graphs,75/25 TRAIN-user split,weighted-LCP and one sampled
  InfoNCE objective;save legacy RQ diagnostics and independent-user metrics.
  Current local probe completed18trials/90snapshots in203.7seconds;no generator
  trained or GPU/Kaggle submission. Artifacts:`/workspace/artifacts/sid-graph-swap`.
  Small positive gain changes do not establish a downstream improvement;paired
  user-bootstrap intervals overlap zero. All earlier Field fits below are
  historical,no pending launches. Preserve paused TPU edits;no Telegram.

- Latest request,2026-10-26:run TIGER on saved GradNorm field-encoder indices.
  Use `experiments/amazon_beauty_field_gradnorm_tiger_2t4.json`;see
  `docs/gr-field-gradnorm-tiger.md`. One private two-T4 session,two generator
  fits on ablation source d13dd8aada6c1f60ce36409147c261af5a09b608,
  gradnorm_s42/43 at AE5000. Reuse exact saved RQ-KMeans codes/books/reports;
  no encoder forward,AE/quantizer fit or nearest-centroid re-export. Preserve
  shared generator/data seed42,4+4/d128,FP16,20epochs,all-prefix protocol,
  validation10/20,NDCG10selection and one FP32beam20test;verify original neural
  and case hashes. Reuse completed AE3500→KMeans TIGER controls rather than
  rerunning them;different AE budgets prevent a pure objective-only claim.
  Field-level encoders are the research focus;title-only is a text-semantics
  control,not the main method and not launched here. No extra controls,tuning,
  automatic retries or Telegram.

- Latest request,2026-10-26:run AE-parameter/reconstruction/MTL curve probes,
  still WITHOUT TIGER. Use `experiments/amazon_beauty_field_ae_ablation_probes_2t4.json`;
  see `docs/gr-field-ae-ablation-probes.md`. One private two-T4 session,nine
  interventions x seeds42/43,each5000fixed updates,same eight checkpoints.
  Variants:latent128,neural LR0.0003,frozen BN running stats(affine trains),
  no-price reconstruction(input retained),cosine title loss,title-only,UW
  weight LR0.03,DWA T2,GradNorm alpha1.5/weightLR0.025. Hold other settings
  fixed;reuse pinned original title inputs,completed fixed-normalized AE curves
  and frozen-T5/RQ-VAE reference. Keep raw-latent3x256KMeans,dimension follows
  AE64/128;initial neural equality is checked within(seed,shape). Save complete
  checkpoints/latents/books/codes,per-field MSE/cosine,TRAIN-only proxies and
  comparison plots. No generator,validation/test,collaborative objective,
  automatic extra tuning/retries or Telegram.

- Latest user clarification,2026-10-26:run fixed AE warm-up stopping probes
  WITHOUT TIGER. Use `experiments/amazon_beauty_field_ae_warmup_probes_2t4.json`;
  see `docs/gr-field-ae-warmup-probes.md`. One private two-T4 session,two
  continuous AE trajectories seeds42/43,5000fixed updates,same field/title
  inputs,latent64,fixed-normalized objective and native Adagrad0.001/FP32.
  Save complete checkpoints and GPU-eval latents at0/100/250/500/1000/2000/
  3500/5000;no adaptive stopping or PCA projection. Fit identical raw-latent
  RQ-KMeans3x256 at each checkpoint and record geometry,reconstruction,usage,
  collisions and catalogue/TRAIN-only first-prefix behavioural proxies.
  Preserve raw arrays,curves,provenance and plots. No generator,TIGER,
  validation/test metrics,collaborative loss,automatic tuning/retries or
  Telegram. The preliminary four-TIGER plan was cancelled before submission.

- The history-compression / Query SSM track was closed on 2026-10-21. Read
  `docs/research/closed-history-compression/README.md` for its final evidence.
- Earlier experiment plans are historical and have no pending GPU launches.
- The active direction is comparing item encoders for SID-based generative
  recommendation. Read `docs/gr-research-brief.md`, `docs/gr-baseline.md`, and
  `docs/research/item-encoders/README.md`.
- Latest user authorization,2026-10-26:run two TIGER fits on AE->RQ-KMeans,
  pipeline seeds42/43,one private two-T4 session. Use
  `experiments/amazon_beauty_field_ae_rqkmeans_tiger_2t4.json`;see
  `docs/gr-field-ae-rqkmeans-tiger.md`. Use byte-pinned original3500-step AE
  catalogue latent arrays from the completed warm-up kernel;no AE/encoder/
  decoder/embedding update or re-export. Fit sequential full-catalogue sklearn
  Lloyd KMeans3x256 on raw64D latents,CPU thread1,n_init1,max_iter300,tol1e-4,
  random_state=AEseed+level,without latent/residual normalization. Then freeze
  the resulting SID index and train only TIGER. Preserve the completed
  AE-freeze generator control's4+4/d128,FP16,20epochs,all131413prefix cases,
  validation10/20,NDCG@10selection and final FP32free-beam20test. Generator/
  data seed42 is shared;match initial neural and case hashes against completed
  control `carasik/rhpr-field-warmup-tiger` source10b53b8. Copy completed
  controls rather than rerunning them. No3k-budget comparison,price ablation,
  automatic tuning/retries or Telegram. Save books,codes,fit diagnostics,
  source AE identity,generator checkpoints/curves/predictions and run metadata.
- Latest user authorization,2026-10-26:compare the completed entire-AE-freeze
  and joint-refinement semantic indices using the same TIGER. Use
  `experiments/amazon_beauty_field_warmup_tiger_2t4.json`;see
  `docs/gr-field-warmup-tiger.md`. One private two-T4 session,four generator
  fits on saved indices ae/joint x quantizer seeds42/43,with generator/data
  seed42 fixed across all four. Byte-pin the completed warm-up parent's source,
  manifest,catalogue order,semantic-code arrays and reports. No encoder,
  quantizer,book initialization or embedding reruns. Preserve early4+4/d128,
  FP16,all131413train-prefix positions per epoch,20epochs,validation10/20,
  NDCG@10 selection and one final full-catalogue free-beam20test per model.
  Keep ordinal suffix/invalid beam accounting unchanged. Check initial neural
  parameter hash against shared CPU control before training;immutable SID
  lookup data intentionally differ. Save best/last,curves,predictions and
  training/index provenance;no study telemetry or automatic retries/tuning.
  Two index seeds are not independent generator-seed repeats. Earlier native
  GPU initializations did not produce bitwise matched third books;this tests
  the saved candidate indices,not a perfectly isolated causal freeze ablation.
- Completed warm-up suite:all six fits successful in12.56minutes. At3x256/
  latent64,full AE freeze produced12004/12018distinct tuples,max group3;
  joint produced11542/11258,max15/17;input-feature freeze11674/11296,max8/14.
  Earlier cold fixed-normalized64D control produced188,max907. Warm-ups both
  stopped3500steps;PCA99%dimension5,nominal24bit active count8. Joint improves
  brand/category/title reconstruction relative to full freeze. Warm-up adds
  compute;this does not isolate codebook lag or establish better NIP yet.
- Latest user authorization,2026-10-26:continuous AE warm-up then native RQ-VAE
  in three modes:joint,entire AE frozen,feature-encoder-only frozen. This
  supersedes the earlier four-arm plan with separate full RQ-KMeans variants.
  Use `experiments/amazon_beauty_field_ae_warmup_rq_2t4.json`;see
  `docs/gr-field-ae-warmup-rq.md`. One private two-T4 session,two pipeline
  seeds42/43,one shared AE per seed,then three5000-main-update RQ branches.
  Transfer embeddings/projections,BN buffers,MLP encoder/decoder,heads and loss
  state. Native mini-batch residual KMeans initializer remains unchanged.
  AE max5000,min2000,500-step catalogue PCA;stop after four equal active-mode
  counts using reverse water-filling at aggregate nominal24bits. This is a
  heuristic adaptation of2605.06870 to3x256 residual books,not a proven bound
  or guarantee of reconstruction convergence. Also log8bit count and PCA99pct.
  Keep fixed-normalized four-field objective,latent64,Adagrad0.001,FP32 and
  batch2048. Reset optimizer accumulators at phase transition,freeze eval modes
  and verify frozen buffers/weights. Save both phases,drift,field losses,usage
  and collisions. No KL,generator,suffix repair,automatic retries or tuning.
- Latest user authorization,2026-10-26:increase RQ latent/code-vector dim64
  to128 with two5000-update runs,fixed normalization and GradNorm,on two T4s.
  Use `experiments/amazon_beauty_field_rq_latent128_2t4.json`;see
  `docs/gr-field-rq-latent128.md`. Keep3x256,input960,hidden widths,LRs,beta,
  initializer,seed42,batch2048 and FP32. Reuse completed64D RQ controls.
  Initial128D hashes must match BETWEEN new arms;do not require the old64D
  hash across different shapes. Log norms divided by sqrt(dim),relative
  quantization MSE,all field metrics,book usage and collisions. No generator,
  codebook-LR change,extra AE fits,suffix repair or automatic tuning/retries.
- Latest user authorization,2026-10-26:restore native RQ-VAE with the completed
  fixed-normalized and GradNorm objectives,one private two-T4 session,two
  independent5000-update quantizer-only trials. Use
  `experiments/amazon_beauty_field_rq_balancing_2t4.json`;see
  `docs/gr-field-rq-balancing.md`. Keep3x256/latent64/native initializer/STE,
  summed codebook+beta0.25commitment at weight1,and neural Adagrad0.001/FP32.
  GradNorm balances only four field losses,not the quantization auxiliary.
  Reuse byte-pinned frozen titles and completed AE controls;verify the same
  initial neural hash before training. Record actual initialization batches
  separately from main reconstruction updates. No generator or suffix repair.
- Latest follow-up:the user requested alternatives to UW and asked about
  available compute. Run one private two-T4 continuous-AE suite using
  `experiments/amazon_beauty_field_ae_nonuw_2t4.json`:equal raw sum,square-root
  constant normalization,DWA(T2) and GradNorm(alpha1.5,weight Adagrad LR0.025),
  5000updates each,same seed42/neural initialization/Adagrad0.001/FP32.
  DWA/GradNorm use the completed fixed-normalized AE as control. Preserve all
  per-field losses/checkpoints/weight diagnostics;no RQ,generator or SID yet.
  See `docs/gr-field-ae-nonuw.md`;do not automatically tune or retry trials.
- Latest user authorization,2026-10-26:check multiple fast continuous-AE loss
  hypotheses,parallel on two T4s. Use
  `experiments/amazon_beauty_field_ae_loss_suite_2t4.json`;four independent
  5000-step trials:UW log-variance LR0.03/0.1 (neural Adagrad LR0.001),fixed
  constant-predictor-normalized losses,and title-only reconstruction with all
  input fields retained. Same seed42,initial neural weights,FP32,latent64 and
  batch2048. Reuse byte-verified frozen title targets and the completed original
  UW control;no baseline rerun,RQ,SID,generator or recommendation claims. See
  `docs/gr-field-ae-loss-suite.md`. This authorizes one private two-GPU session,
  not arbitrary follow-up tuning or automatic retries.
- Latest user authorization,2026-10-26:run the separately trained continuous
  fields+frozen-title autoencoder control,5000updates,seed42,same native neural
  initialization/latent64/Adagrad1e-3/uncertainty objective. Use
  `experiments/amazon_beauty_field_title_ae_5k_control.json`;see
  `docs/gr-field-autoencoder-control.md`. Remove both quantized forward and
  codebook/commitment loss;no SID,codebook initialization or generator. Compare
  every field at500-step catalogue evaluations,not only weighted total loss.
- Latest follow-up,2026-10-26:the user requested a5000-step extension of the
  fields+frozen-title QUANTIZER-ONLY arm. Use
  `experiments/amazon_beauty_field_title_rqvae_5k_diagnostic.json` with identical
  architecture,optimizer,seed42,objective and500-step diagnostics. Fresh total
  5000updates,not5000extra and not automatic checkpoint resume. Depth remains
  unchanged;the user's deeper-network suggestion is a future separate ablation.
- Latest user decision,2026-10-26:two QUANTIZER-ONLY3000-step GPU runs to compare
  ordinary Sentence-T5(full metadata)RQ-VAE with the joint structured encoder
  plus frozen Sentence-T5(title only). Use
  `experiments/amazon_beauty_rqvae_plain_3k_diagnostic.json` and
  `experiments/amazon_beauty_field_title_rqvae_3k_diagnostic.json`. Title768D
  concatenates directly with learned192D fields;fourth reconstruction head MSE
  joins uncertainty weighting. Same seed42,3x256,latent64,native optimizer and
  catalogue diagnostics every500steps. Explicitly no generator/validation/test.
  Suffix overflow is a recorded measurement in this diagnostic stage,not an
  execution failure;never repair/widen it. This supersedes initial title exclusion
  for the new arm only. Read `docs/gr-rqvae-title-comparison.md`.
- Both3kdiagnostic runs completed:plain11991tuples/max group4/used143,256,256;
  fields+title845tuples/max1162/used10,42,157,still suffix overflow. The plain
  run has13stdout events and7full probe snapshots;its console summary was
  shorter,not absent. Future plain logs now expose matching summary fields.
  Do not retrain to recover these artifacts or call title a complete collapse fix.
- The user specified joint feature-encoder/RQ-VAE reconstruction on2026-10-26:
  brand embedding+Linear/categories MeanPooling embeddings+Linear/standardized
  price+Linear,concatenate into native GRID RQ-VAE;CE/BCE/Huber combined by
  LibMTL uncertainty weighting. Title is explicitly excluded. Read
  `docs/gr-field-rqvae.md`;config
  `experiments/amazon_beauty_field_rqvae_tiger_fast_prefix.json`. Preserve the
  fast generator/training/evaluation control. This metadata objective is not
  joint NIP training and does not establish that text encoders are inferior.
- The first field-RQ-VAE run failed at SID suffix overflow after3000updates:
  68unique semantic IDs for12101items,largest group1796,used codes5/8/13.
  The user requested5000updates and codebook diagnostics on2026-10-26. Use
  `experiments/amazon_beauty_field_rqvae_tiger_5k.json`:fresh same-seed fit,
  full-catalogue eval probes after initialization/every500steps/final,counts,
  entropy/perplexity,residual/centroid geometry,raw loss components and snapshots.
  Preserve architecture,optimizer,uncertainty,fast generator and suffix policy;
  do not silently widen the suffix or reinitialize dead codes.
- The user authorized a fast Sentence-T5 + native GRID RQ-VAE3x256 + previous
  GRID TIGER4+4/d128 baseline on2026-10-25:3000 quantizer updates,20 generator
  epochs,FP16 generator,validation every10,no study telemetry. Read
  `docs/gr-rqvae-baseline.md`. This is separate from joint encoder training;
  do not resume the paused index-study kernels.
- The user authorized restoring the faster early prefix protocol on2026-10-25:
  sampling is not the item-encoder research axis. Use
  `experiments/amazon_beauty_rqvae_tiger_fast_prefix_fp16_20epochs.json`.
  Restore the early4+4/d128 ordinary-FF generator: noSEP,user bins2000,
  4,977,920parameters,beam20,Adam1e-3/wd1e-4/clip1,batch256/physical128.
  Every noninitial train position appears once per epoch, latest20 context,
  131,413examples. Preserve RQ-VAE3x256/3000updates, frozen Sentence-T5,
  FP16 generator,20epochs, validation only at10and20, NDCG@10 selection.
  The expanded-FF and plain-FF sampled-window configs are historical;
  do not launch them as the active baseline or revive the paused studies.
- The SID index-structure investigation was paused on 2026-10-25 at the user's
  request. Read `docs/research/paused-sid-index/README.md` and its results ledger
  before reopening it. It did not establish a causal semantic-collision rule
  or prove text encoders are defective; SHANY primarily exposed missing brand
  metadata. The historical study (2026-10-24) used six RQ-KMeans seeds,
  frozen Sentence-T5 embeddings, identical generator initialization and shared
  TIGER/GRID window manifests. Two T4s in one session, one independent seed
  worker per GPU (three seeds each), FP16, 3+3 layers at d128, 20 epochs,
  validation every5, no early stopping, final-epoch comparison. Read
  `docs/gr-sid-index-study.md`. No UI: Parquet evidence with optional DuckDB.
- Raw autoregressive scores and valid-child locally renormalized scores are
  separate objectives. Compare exact search and beams under MATCHING scoring.
  Do not interpret six index seeds as six independent generator seeds.
- The completed follow-up crossed frozen original rq11/rq22 indices with
  generator seeds101/202/303, preserving data seed4242 and20epochs. See
  `docs/gr-sid-training-variance.md`. Fixed train/validation probes preserve
  RNG/weights; their Parquet sidecars are distinct from checkpoint evaluations.
  Initialization hashes must match WITHIN each model seed across indices.
- The user explicitly wants GPU, not TPU; implementation does not authorize
  dispatch. Preserve the paused TPU changes; do not restart previous kernels.
- Historical reproduction work is in `docs/gr-genpas-tiger-epochs.md`,
  `docs/gr-genpas-tpu.md`, `docs/gr-grid-quality.md` and
  `docs/gr-grid-reproduction.md`. Do not silently revive those protocols.
- Do not implement unrequested tokenizers or assert a representation bottleneck
  before the baseline and diagnostics support that claim.

## Working agreement

- Keep experiments reproducible: configuration, seed, code revision, and final
  scalar metrics belong in the run output.
- Never commit credentials, `.env` files, downloaded private data, or Kaggle
  API tokens.
- Work on a branch and prefer a pull request over writing directly to `main`.
- Run `python3 -m unittest discover -s tests -v` before publishing changes.
- Do not spend Kaggle quota unless the user explicitly asks to run or deploy an
  experiment. Code changes alone are not authorization to start compute.

## Experiment contract

- Configurations live under `experiments/` as JSON.
- `kaggle/run.py` is the stable kernel entry point.
- Successful runs write `/kaggle/working/metrics.json` and
  `/kaggle/working/summary.md`.
- `metrics.json` must be a flat JSON object of finite scalar values. Add richer
  outputs as separate artifact files.
- Keep Kaggle kernels private unless the user explicitly requests publication.

## Integrations

- Use the GitHub connector/MCP for repository, PR, workflow, and run status
  operations. Use `gh` only when the connector does not expose the needed
  operation.
- Use Kaggle MCP for quota/status/output inspection and interactive control.
  Use `.github/workflows/kaggle-experiment.yml` for reproducible execution.
- Telegram is an outbound notification channel only. Send lifecycle summaries
  and workflow links; never send secrets, datasets, stack traces, or full logs.
- When MCP and CLI report different state, treat Kaggle/GitHub server state as
  authoritative and record the discrepancy.

The repo-local `rhpr-ops` plugin contains detailed GitHub, Kaggle, and Telegram
skills. Use the relevant skill whenever a request touches those integrations.
