closeAllConnections()
rm(list=ls())
#
# source("/home/betancurj/dev/ML_RevascCathCohort/GlobalVars.R")        # control vars
source("C:/dev/ML_RevascCathCohort/GlobalVars.R") 
#
setwd(workdirectory)
#
#############
# LIBRARIES #
#############
if (!require("data.table", character.only=TRUE)){ install.packages("data.table") }
if (!require("xtable", character.only=TRUE)){ install.packages("xtable") }
if (!require("rJava", character.only=TRUE)){ install.packages("rJava") }
if (!require("RWeka", character.only=TRUE)){ install.packages("RWeka") }
if (!require("plyr", character.only=TRUE)){ install.packages("plyr") }
if (!require("pROC", character.only=TRUE)){ install.packages("pROC") }
if (!require("FSelector", character.only=TRUE)){ install.packages("FSelector") }
if (!require("PredictABEL", character.only=TRUE)){ install.packages("PredictABEL") }
if (!require("e1071", character.only=TRUE)){ install.packages("e1071") }
#
library(data.table)
library(xtable)
library("RWeka") # Use wrapper WEKA
library(plyr)  # count
library(pROC)
library(FSelector) # information gain
library(base) # switch function
#
library(caret) # stratified cross-validation
library(PredictABEL) # NRI
library(e1071) # SVM
#
################################ FUNCTIONS #####################################
#
## return idxs of lines to be used in folds. 
# params is a frame with at leat one variable coding the "outcome" used to stratify the CV is kFold is used
# Data has to have a column with outcome given in params
# if type == "variable", params has to have a variable "site" with the columname of the variable to use to stratify per site or otr
CreateFoldsPerVariable <- function(type, column,dataIn, outcomeNamePatient="R90", seed=0, k=10){
  # Creates folds following levels of column.
  if(column=="patid"){ ### 
    set.seed(seed)
    dataPat <- dataIn[ !duplicated(dataIn$patid),  ]
    foldsPat <- createFolds(dataML[,outcomeNamePatient], k = 10) # stratified CV
    
    folds <- list()
    
    for (fold in 1:k ){
      # idx <- foldsPat[[fold]]
      folds[[fold]] <- which( dataIn$patid %in% dataPat$patid[ foldsPat[[fold]] ] )
    }
    names(folds) <- 1:k
  }
  else{
    print(column)
    varDict <- dataIn[!duplicated(dataIn[,column]),column]
    print(varDict)
    
    folds <- list()
    for (v in varDict ){
      folds[[v]] <- which( dataIn[,column] == v )
    }
  }
  
  
  return(folds)
}
#
#save test datasets #
##########################################################  VARIABLES ##########################################################################################
####
# Input data
####
#
n_outcome <- 1
for ( outcomeName in outcomeNameVector ){
  dir.create(outputDir[n_outcome])
  dataML <- read.csv( file=dataFileName[n_outcome], stringsAsFactors = FALSE, check.names = FALSE)# entire clean dataset
  
  #
  #dataML <- dataML[which(dataML$`suspectCADwith stress image` == 1) ,]
  
  # if(isR90){
  #   dataML <- dataML[which(!is.na(dataML$`R90`) ),]
  # }else{
  #   ## 180 revasc
  #   dataML <- dataML[which(!is.na(dataML$`R180`) ),]
  # }
  # 
  # if (isR90_R180){
  #   dataML <- dataML[which(!is.na(dataML$`R90`) & !is.na(dataML$`R180`) ),]
  # }
  # idx_isch <- which(!is.na(dataML$isch_tpd.View1))
  # #
  #
  #########
  # dataML <- read.arff("NXTPlaqueML feb18-2016.arff")
  #
  # file with all information about variables
  vars <-read.csv(varsFilename[n_outcome], check.names = FALSE, stringsAsFactors = F)
  #
  # # converting data types and imputing NAs 
  idx<-1
  for (idx in 1:length(vars$name)){
    # characters always as factors. "" considered as NAs and imputed
    name<-vars$name[idx] 
    if (name %in% names(dataML)){
      
      if(vars$dataType[idx] == "double"){
        print(name)
        if (typeof(dataML[,name]) == "character"){
          dataML[,name] <- ifelse( dataML[,name] == "", NA, dataML[,name] )
        }
        dataML[,name] <- as.double(dataML[,name])
        dataML[,name] <- ifelse(is.na(dataML[,name]), ifelse(imputeNAs, mean( dataML[,name], na.rm=TRUE), NA),  dataML[,name])
      }
      #
      ## imputing other types
      #
      if (typeof(dataML[,name]) == "character"){
        dataML[,name] <- ifelse( dataML[,name] == "", ifelse(imputeNAs, "NA", NA), dataML[,name] )
        dataML[,name] <- as.factor( dataML[,name] ) # exlude = NULL forces NA to be another level
      }
      #
      # these must be nominal (list in variables section)
      if (vars$dataType[idx] == "factor"){
        # NAs replaced by NA string
        dataML[,name] <- ifelse(is.na(dataML[,name]), ifelse(imputeNAs, "NA", NA), as.character( dataML[,name])) # needed to get the WEKA implementation of logitBoost to work
        # convert to factor
        dataML[,name] <- as.factor(dataML[,name])
      }
      #
      # for doubles we impute NAs using mean value
      if (typeof(dataML[,name]) == "double"){
        dataML[,name] <- ifelse(is.na(dataML[,name]), ifelse(imputeNAs, mean( dataML[,name], na.rm=TRUE), NA),  dataML[,name])
        print( paste(name,": ", mean( dataML[,name]) ) )
      }
      #
      ###for integers, we convert them to nominals
      if (typeof(dataML[,name]) == "integer"){ 
        # dataML[,name] <- ifelse(is.na(dataML[,name]), ifelse(imputeNAs,max(dataML[,name], na.rm=TRUE), NA),dataML[,name])
        dataML[,name] <- as.factor(as.character(dataML[,name])) # exlude = NULL forces NA to be another level
      }
      #
      # forcing  variables to be numeric according to input variables description file
    }else
    {
      print(paste(name, " not in input  data."))
    }
  }
  #   
  ## filtering by age
  # dataML_cpy<-dataML
  #########
  #
  write.csv(dataML,file=paste(outputDir[n_outcome],"DataConvertedImputed.csv",sep=""), row.names = FALSE)
  write.arff(x=dataML,file=paste(outputDir[n_outcome], "DataConvertedImputed.arff", sep=""))
  #
  typesData<-c();for (name in names(dataML)){print(typeof(dataML[,name]));typesData<-rbind(typesData,typeof(dataML[,name]))}
  #
  ### read vars filename
  # vars <- as.data.frame(t(read.csv(varsFilename, header=FALSE)) )
  # names(vars) <- c("name","DataType","Category")
  # row.names(vars) <- NULL 
  # vars$name <- as.character(vars$name )
  #
  patid <- vars$name [which(vars$Category == "ID")] # patid
  #
  # dataML_cpy <- dataML[,c(patid, outcomeNameVector)]; # TODO: copy only outcomes and patid
  dataML_cpy <- dataML
  #
  xRefAllTest <- data.frame(fileName=as.character(),
                            test=as.character(),
                            outcome=as.character(), 
                            cvType = as.character(), 
                            followUp=as.integer()) # master frame with info to do plots
  #
  ## Creating folds for cross-validation.. use same folds. 
  #
  #### 
  for (seed in seeds){
   
     if (createFoldsOn){
      # for (outcomeName in outcomeNameVector){
        for (cv in cvType){
          set.seed(seed)
          folds <- createFolds(dataML[,outcomeName], k = 10) # stratified CV
          if (cv != "kFold"){
            folds <- CreateFoldsPerVariable("variable", cv, dataML, outcomeName)
          }
          filename <- paste(outputDir[n_outcome],"Folds_",cv,"_",outcomeName,"_seed_",seed,".RData",sep="")
          save(folds, file=filename)
        }
      # }
     }
    ### 
    
    for (fu in followup){
      ### merging original events
      # dataML <- dataML[,which(!(names(dataML) %in% c(outcomeNameVector)))]
      # dataML <- merge(dataML, dataML_cpy, by=patid)
      dataML <- dataML_cpy
      
      ### censoring events beyond max follow-up
      if (!is.na(fu)){
        dataML[which(dataML[,fuVar] > fu), outcomeNameVector] <- 0 
      }
     
      # #####################
      # #save test datasets #
      # #####################
      if( saveTestsData ){
        toFilter <- c(patid, outcomeNameVector) # features to filter from Machine learning. Must include outcome
        idxExp <- 1
        # for ( idxExp in 1:dim( nameExp[which(nameExp$outcome == outcomeName),] )[1] )
        for ( idxExp in which(nameExp$outcome == outcomeName) )
        {
          # features to use in the experiment
          toUse <- VarsExperiment(nameExp[idxExp,"nameExperiment"], vars)
          #
          #save test datasets #
          write.arff(file=paste(outputDir[n_outcome],nameExp[idxExp,"nameExperiment"],"_fu-",fu,".arff", sep=""), dataML[,toUse])
          write.csv(file=paste(outputDir[n_outcome],nameExp[idxExp,"nameExperiment"],"_fu-",fu,".csv", sep=""), dataML[,toUse])
        }
      }
      
      ### TESTS!  ------ check, now this loop is the outer loop
     # for ( outcomeName in outcomeNameVector ){
        #
        toFilter <- c(patid, outcomeName) # features to filter from Machine learning. Must include outcome
        mlOutcome <- data.frame(patid=dataML[,patid] )# variable with experiment results
        idxExp <- 1
        #
        # Test following var category
        # for ( idxExp in 1:length(nameExp) )
        for ( idxExp in which(nameExp$outcome == outcomeName) )
        {
          expDir <- paste(outputDir[n_outcome],outcomeName,"_fu-",fu,"_seed_",seed,"_",nameExp[idxExp,"nameExperiment"], sep="")
          filenameSuffix <-paste(outcomeName,"_fu-",fu,"_",nameExp[idxExp,"nameExperiment"],sep="")
          unlink(expDir, recursive=TRUE) # clear directory
          dir.create(expDir)
  
          toUse <- VarsExperiment(nameExp[idxExp,"nameExperiment"], vars)
          ####################
          #
          #################### 
          # Machine learning #
          ####################
          #
          # Cross Validation loop
          start.time <- Sys.time()
          for (cv in cvType){
            # loading folds 
            if (exists("folds")){
              rm (folds)
            }
            fileFolds <- paste(outputDir[n_outcome],"Folds_",cv,"_",outcomeName,"_seed_",seed,".RData",sep="")
            load(file=fileFolds, envir = parent.frame(), verbose = FALSE)
            
            K <- length(folds) # number of folds
            
            predictionCV <- c() # prediction LogitBoost model
            predictionCV_SVM <- c() # prediction SVM model
            predictionCV_LR <- c() # only weka implementation is supported
            
            framesCV <- c() # feature selection main CV
            
            
            for (idx in seq(1,length(folds))){
              
              train <- c()
              for (k in seq(1,K)){
                if (k != idx){
                  train <- rbind(train, dataML[folds[[k]],toUse] )
                }
              }
              test <- dataML[folds[[idx]],toUse] 
              #
              ## save folds
              if (saveCsvFolds){
                write.csv(file=paste(outputDir[n_outcome],"/Fold_",idx,"_Test.csv", sep=""), x=test, row.names=FALSE)
                write.csv(file=paste(outputDir[n_outcome],"/Fold_",idx,"_Train.csv",sep=""), x=train, row.names=FALSE)
              }
              ##############################
              #
              # Summary fold data
              print(paste("CV type: ", cv))
              print(paste("Fold: ",  names(folds)[idx]))
              print(paste("outcome: ", outcomeName, ", follow-up: ", fu))
              print(paste("Train: ", outcomeName, " = ", length(which(train[outcomeName]==1)),"/",length(train[,outcomeName])))
              print(paste("Test: ", outcomeName, " = ", length(which(test[outcomeName]==1)),"/",length(test[,outcomeName])))
              # summary(train)
              #
              selected<- which(!(names(train) %in% toFilter))
              #
              if(featureSelection){
                # FEATURE SELECTION
                selected <- c()
                
                selectedFrame <- data.frame(name=character(), merit=double(), sd.merit=double(), stringsAsFactors = FALSE)
                frames <- data.frame(name=character(), merit=double(), sd.merit=double(), stringsAsFactors = FALSE)
               
                # CV for feature selection
                nFoldsFS <- 1
                if (cvFeatureSelection){
                  nFoldsFS <- 5
                  foldsFS <-createFolds(train[, outcomeName], k = nFoldsFS, list = FALSE) # stratified cross-validation
                }else{
                  foldsFS <- rep(2,dim(train)[1]) # stratified cross-validation. NONE by default
                }
                
                # information gain per feature in train set
                for (jdx in seq(1,length(names(train)))){
                  feature <- names(train)[jdx]
                  if ( !(feature %in% toFilter) ){
                    infoGain <-c()
                    for(fold in 1:nFoldsFS){ # cross validation to compute merit
                      
                      if (isWEKAML){
                        # WEKA feature selection algorithm implementation reached by RWeka
                         # infoGain <- c( infoGain, InfoGainAttributeEval(as.formula(paste(outcomeName," ~ .",sep="")),  data = train[which(foldsFS != fold),c(feature,outcomeName)]) ) # Information gain
                        infoGain <- c( infoGain, GainRatioAttributeEval( as.formula(paste(outcomeName," ~ .",sep="")), data = train[which(foldsFS != fold), c(feature,outcomeName)]) ) # Information gain ratio. Avoids bias due to higher feature "values"
                      }else{
                        #R feature selection algorithm implementations
                         infoGain <- c( infoGain, information.gain(as.formula(paste(outcomeName," ~ .",sep="")), data = train[which(foldsFS != fold),c(feature,outcomeName)]) )
                        # infoGain <- c( infoGain, gain.ratio(as.formula(paste(outcomeName," ~ .",sep="")), data = train[which(foldsFS != fold),c(feature,outcomeName)]) )
                      }
                    }
                    
                    # selection of those with infoGain > 0.0
                    if (mean(infoGain) > 0.0 ){ # Features with info gain > 0.0 using InfoGainAttributeEval from WEKA
                      # selected <- c(selected,jdx)
                      selectedFrame <- rbind(selectedFrame, data.frame(name=names(infoGain[1]), merit=mean(infoGain), sd.merit=sd(infoGain) ))
                    }
                    frames <-  rbind(frames, data.frame(name=names(infoGain[1]), merit=mean(infoGain), sd.merit=sd(infoGain) )) ## for all features
                    # cat("infoGain Feature ", feature, " ", mean(infoGain), " +- ", sd(infoGain),"\n")
                    # summary(infoGain)
                  }
                }
                tempFramesCV <- frames
                tempFramesCV$NFolds <- nFoldsFS
                tempFramesCV$Fold <- idx
                framesCV <- rbind( framesCV, tempFramesCV) 
                
                frames <- frames[order(frames$merit, decreasing=TRUE),]
                
                write.csv(file=paste(expDir,"/",cv,"_", names(folds)[idx], "_featureSelection.csv",sep=""), frames)
                
                
                selectedFrame <- selectedFrame[order(selectedFrame$merit, decreasing=TRUE),] # ranking features by merit (highest to lowest)
                selectedFrame$AUC <- NA # AUC variable for feature selection
                selectedFrame$rank <- 1:dim(selectedFrame)[1]
                
                cat("****** Feature selection, fold ", idx, " ***************\n")
                print(selectedFrame)
                selected <- which( names(train) %in% selectedFrame$name ) # features in train dataset with info gain > 0.0
                
            
              }
              #
              # #### LOGISTIC REGRESSION ####
              # if (includeLogistic){
              #   modelLR <- stats::glm( as.formula(paste(outcomeName," ~ .",sep="")), data=dataSelected, family=binomial)
              # }
              # summary(modelLR)
              # predTestLR <- stats::predict(modelLR, newdata=test[,selected], type='response')
              # names(predTestLR) <- NULL
              # 
              # predTestLR <- data.frame( LR= predTestLR, patid = test[,patid], outcome = test[, outcomeName]  )
              # predFoldLR <- predTestLR
              ### TODO: include LR
              #
              ####################
              # LogitBoost model #
              ####################
              if (isWEKAML){
                trainSelected <- train[, c(selected, which(names(train) == outcomeName))]# selected train vars and outcome variable
               
                 modelLB <- RWeka::LogitBoost( as.formula(paste(outcomeName," ~ .",sep="")),  data = trainSelected, control = controlLB )
                 .jcache(modelLB$classifier) 
                 #
                 save(modelLB ,file=paste(expDir,"/", cv,"_Fold_", names(folds)[idx],"_",nameExp[idxExp,"nameExperiment"],"_modelWEKA.RData",sep="") )
                 
                 writeLines(modelLB$classifier$toString(), paste(expDir,"/", cv,"_Fold_", names(folds)[idx],"_",nameExp[idxExp,"nameExperiment"],"_modelWEKA.txt",sep=""))
                 
                 
                 #
                 print("LogitBoost model model built in WEKA")
                 # names(train[,selected]) == names(test[,selected])
                 predTest <-data.frame( predict(modelLB, newdata=test[,selected], type=c("probability")), patid = test[,patid], outcome = test[, outcomeName]  )
                 predFold <- predTest 
              
              } else{
                # if (nameExp[idxExp] == "LogitBoost"){
                  
                  print("Model built in R")
                  # R implementation of logitBoost
                  # selectedMC <- c(selected, which(names(train) == "MACE")) # Adding MACE feature
                  # dataSelected <- train[, selectedMC]
                  modelLB = caTools::LogitBoost(train[,selected], train[,outcomeName], nIter=50)
                  # modelLB = ada::LogitBoost(train[,c(3,4)], train[,outcomeName], nIter=50)
                  # modelLB <- ada::ada(as.formula(paste(outcomeName," ~ .",sep="")), data=train[, c(selected, which(names(train) == outcomeName))],loss="logistic", iter=50)
                  # modelLB <- update(modelLB, x=train[, selected], y = train[,"MACE"], test.x=test[,selected], test.y=test$MACE,loss="logistic", n.iter=50)# update model to enable prediction of unseen test observations
                  # predict test data
                  p = predict(modelLB, newdata = test[,selected], type="probs")
                  predFold <- data.frame(X0=p[,1], X1 = p[,2],  patid = test[,patid], outcome = test[,outcomeName])
                  rm(p)
              }
              #
              predictionCV <- rbind(predictionCV, predFold) # for final ROC
              #
              # ROC for fold predictions LogitBoost
              predictorTest <- nameExp[idxExp,"nameExperiment"]
              p <- "X1" # prob for MACE=1
              predFold[,p] <- round(as.numeric(predFold[,p]),4) 
              labelPredictor <- paste("LogitBoost, ", predictorTest, sep="")
              t1 = data.frame(Predictor=p, AUC=NA, L95=NA, U95=NA)
              #
              roc2 = pROC::roc(as.formula(paste("outcome"," ~", p)), data=predFold)
              t1[t1$Predictor==p,c("L95","AUC","U95")] = round(as.numeric(pROC::ci(roc2)),2)
              #
              t =   paste(labelPredictor, ": ", round(t1$AUC,2)," (", round(t1$L95,2)," to ",round(t1$U95,2),")", sep="")
              #
              pdf(paste(expDir,"/", cv,"_Fold_", names(folds)[idx],"_",nameExp[idxExp,"nameExperiment"],"_Test.pdf",sep=""), width = 8.5, height = 8.5, pointsize = 14)
              #
              # title <- paste(outcomeName, " prediction Fold ", names(folds)[idx]," with ",nameExp[idxExp,"nameExperiment"], sep="" )
              title <- paste(outcomeName, " (fu= ", fu, " days) by ", nameExp[idxExp,"nameExperiment"], " (Fold ", names(folds)[idx],")", 
                             "\n(",length(predFold$outcome[which(predFold$outcome==1)]) , "/", length(predFold$outcome)," events)", sep="")
              #
              invisible(plot(roc(as.formula(paste("outcome"," ~", p)),data=predFold,xlim=c(0,1),ylim=c(0,0.99),smooth=FALSE), main=title))
              legend(0.85,0.5,t,lty=1,lwd=2,cex=1.1,box.lty=0, bty= "n" ,title="Method, Test: AUC (95% CI)")
              #
              dev.off()
              
              ## LOGISTIC:
              if (includeLogistic){
                if (isWEKAML){
                  trainSelected <- train[, c(selected, which(names(train) == outcomeName))]# selected train vars and outcome variable
                  #WEKA
                  modelLB <- RWeka::Logistic( as.formula(paste(outcomeName," ~ .",sep="")),  data = trainSelected, control = controlLR )
                  print("Logistic model model built in WEKA")
                  names(train[,selected]) == names(test[,selected])
                  predTest <-data.frame( predict(modelLB, newdata=test[,selected], type=c("probability")), patid = test[,patid], outcome = test[, outcomeName]  )
                  predFold <- predTest 
                }
                else{
                    ### TODO: include LR
                    # #### LOGISTIC REGRESSION ####
                    # if (includeLogistic){
                    #   modelLR <- stats::glm( as.formula(paste(outcomeName," ~ .",sep="")), data=dataSelected, family=binomial)
                    # }
                    # summary(modelLR)
                    # predTestLR <- stats::predict(modelLR, newdata=test[,selected], type='response')
                    # names(predTestLR) <- NULL
                    # 
                    # predTestLR <- data.frame( LR= predTestLR, patid = test[,patid], outcome = test[, outcomeName]  )
                    # predFoldLR <- predTestLR
                     predFold <- data.frame(X0=rep(0.0,dim(test)[1]), X1=rep(0.0,dim(test)[1]), patid = test[,patid], outcome = test[, outcomeName] )
                }
                predictionCV_LR <- rbind(predictionCV_LR, predFold) # for final ROC
                
              }
              ## SVM
              if (includeSVM){
                ###########################
                # SVM model (using e1071) #
                ###########################
                # TODO: resolve discrepancy in AUC vs. training in WEKA (lower for e10)
                classWeights=list(c("0"=length(which(train[,outcomeName] == 1))/dim(train)[1], 
                                    "1"=length(which(train[,outcomeName] == 0))/dim(train)[1] ))# class weights
                cat("tunning SVM model\n")
                
                tunecontrolSVM<-tune.control(sampling="cross", #"fix", # 
                                             # fix=2/3, # whole train set used to get best params
                                             cross=3, 
                                             performances = TRUE, 
                                             best.model=TRUE, 
                                             error.fun=function(true.y, pred){                          # error function is auc
                                               pred<-as.data.frame(attr(pred,"probabilities"));
                                               auc<-pROC::roc(response=true.y, predictor=pred[,"1"])$auc[1];  
                                               cat(auc,",", 1-auc," - ")                                # verbose fitness
                                               return(1.0-auc) # tune looks for the lowest error... so, fitness is the difference to ideal AUC
                                             }
                )
                #
                predictFun=function(model,x){ predict(model,x,probability=TRUE)} # prediction function is probabilities
                #
                tuned <- tune.svm(x=as.formula(paste(outcomeName, " ~ .")), 
                                  data=train[,c(selected, which(names(train)==outcomeName))], 
                                  type='C-classification', 
                                  kernel="radial", 
                                  gamma=gamma_, 
                                  cost=cost_, 
                                  class.weights=classWeights, 
                                  probability=TRUE, 
                                  predict.fun=predictFun, 
                                  tunecontrol=tunecontrolSVM )
                # summary(tuned)
                cat("\n")
                print( paste("Training Fitness best model:", tuned$best.performance) )
                #
                a <- predict( tuned$best.model, newdata=test[, selected],probability=TRUE )
                probs <- as.data.frame( attr(a,"probabilities") )
                names(probs) <- paste( "predSVM_", names(probs), sep="" )
                # probs$patid <- test$patid
                #
                predFoldSVM <- data.frame(X0=probs[,1], X1 = probs[,2],  patid = test[,patid], outcome = test[,outcomeName])
                #
                predictionCV_SVM <- rbind( predictionCV_SVM, predFoldSVM ) # stacked SVM predictions
                #
                #
                # ROC for fold predictions LogitBoost
                predictorTest <- nameExp[idxExp,"nameExperiment"]
                p <- "X1" # prob for MACE=1
                labelPredictor <- paste( "SVM, ", nameExp[idxExp,"nameExperiment"])
                t1 = data.frame(Predictor=p, AUC=NA, L95=NA, U95=NA)
                #
                roc2 = pROC::roc(as.formula(paste("outcome"," ~", p)), data=predFoldSVM)
                t1[t1$Predictor==p,c("L95","AUC","U95")] = round(as.numeric(pROC::ci(roc2)),2)
                #
                t =   paste(labelPredictor, ": ", round(t1$AUC,2)," (", round(t1$L95,2)," to ",round(t1$U95,2),")", sep="")
                #
                pdf(paste(expDir,"/", cv,"_Fold_", names(folds)[idx],"_",nameExp[idxExp,"nameExperiment"],"_SVM_Test.pdf",sep=""), width = 8.5, height = 8.5, pointsize = 14)
                #
                # title <- paste(outcomeName, " prediction Fold ", names(folds)[idx]," with ",nameExp[idxExp], sep="" )
                title <- paste(outcomeName, " (fu= ", fu, " days) by SVM (Fold ", names(folds)[idx],")", 
                               "\n(",length(predFold$outcome[which(predFold$outcome==1)]) , "/", length(predFold$outcome)," events)", sep="")
                #
                invisible(plot(roc(as.formula(paste("outcome"," ~", p)),data=predFold,xlim=c(0,1),ylim=c(0,0.99),smooth=FALSE), main=title))
                legend(0.85,0.5,t,lty=1,lwd=2,cex=1.1,box.lty=0, bty= "n" ,title="Method, Test: AUC (95% CI)")
                #
                dev.off()
                
                
              }
              #
            }# end for cross-validation
            #
            # Saving information from feature selection over folds.        
            if(featureSelection){
              # tempFramesCV <- framesCV
              framesCV <- framesCV[,c("name", "merit","Fold")]
              framesCV <- reshape(framesCV, timevar="Fold", idvar="name", direction="wide")
              framesCV$mean <- apply(framesCV[2:K+1],1,mean)
              framesCV$sd <- apply(framesCV[2:K+1],1,sd)
              framesCV <- framesCV[order(framesCV$mean, decreasing=TRUE),] 
              
              write.csv(file=paste(outputDir[n_outcome],"FeatureSelection_",filenameSuffix,"_",cv,".csv",sep=""), framesCV)
            
              #plot feature selection
              tempVars <- vars
              framesCV <- merge(framesCV, vars[,c("name","Category")], by="name")
              framesCV <- framesCV[,c("name","mean","sd","Category")]
             
              # rownames(tempPlot) <- 1:nrow(tempPlot)
              ggplot(framesCV, aes(x = reorder(name, mean), y=mean,fill=Category)) + geom_bar(stat="identity") + coord_flip() +
                ylab("Information gain ratio") + xlab("Variable") + 
                labs(title = "Variables ranking")
          
              ggsave(paste(outputDir[n_outcome],"FeatureSelection_",filenameSuffix,"_",cv,".pdf",sep=""), 
                     plot = last_plot(), 
                     device = NULL, 
                     path = NULL, 
                     scale = 1, 
                     width = 20, 
                     height = 40, 
                     pointsize = 12, 
                     units = "cm")
        
            }
            # ROC for CV predictions 
            filenamePrediction <- paste( "predictions_",filenameSuffix,"_",cv,".csv",sep="" )
            write.csv(file=paste(outputDir[n_outcome],filenamePrediction,sep=""), predictionCV, row.names=FALSE )
            xRefAllTest <- rbind(xRefAllTest, data.frame(fileName=filenamePrediction,
                                                         test=paste("ML ",nameExp[idxExp,"nameExperiment"],sep=""),
                                                                 outcome=outcomeName, 
                                                                 cvType=cv,
                                                                 followUp=fu
                                                                 )
                                     )
            
            p <- "X1" # prob for MACE=1
            # predictionCV[,p] <- round(as.numeric(predictionCV[,p]),3) 
            labelPredictor <- "LogitBoost"
            t1 = data.frame(Predictor=p, AUC=NA, L95=NA, U95=NA)
            
            roc2 = pROC::roc(as.formula(paste("outcome"," ~", p)), data=predictionCV)
            t1[t1$Predictor==p,c("L95","AUC","U95")] = round(as.numeric(pROC::ci(roc2)),3)
            
            t =   paste(labelPredictor, ": ", round(t1$AUC,3)," (", round(t1$L95,3)," to ",round(t1$U95,3),")", sep="")
            
            pdf(paste(outputDir[n_outcome],"ROC_",filenameSuffix,"_",cv,"_seed-",seed,".pdf",sep=""), width = 6.5, height = 6.5, pointsize = 14)
            
            title <- paste(outcomeName, " (fu= ", fu," days) by ", nameExp[idxExp,"nameExperiment"], " (CV: ",cv,")", 
                           "\n(",length(predictionCV$outcome[which(predictionCV$outcome==1)]) , "/", length(predictionCV$outcome)," events)", sep="")
  
            plot(roc(as.formula(paste("outcome"," ~", p)),data=predictionCV,xlim=c(0,1),ylim=c(0,0.99),smooth=FALSE), main=title)
            legend(0.75,0.5,t,lty=1,lwd=2,cex=1.1,box.lty=0, bty= "n" ,title="AUC (95% CI)")
            
            dev.off()
            
            # stacking with experiements variable
            names(predictionCV)[which(names(predictionCV) == p)] <- paste("predLogitBoost_",nameExp[idxExp,"nameExperiment"],sep="") # predictor name changed to experiment name
            # names(predictionCV)[which(names(predictionCV) == "patid")] <- patid 
            mlOutcome <- merge(mlOutcome,predictionCV[,c("patid",paste("predLogitBoost_",nameExp[idxExp,"nameExperiment"],sep=""))], by="patid")
            
            if(includeLogistic){
              names(predictionCV_LR)[which(names(predictionCV_LR) == p)] <- paste("predLogRegression_",nameExp[idxExp],sep="") # predictor name changed to experiment name
              # names(predictionCV_LR)[which(names(predictionCV_LR) == "patid")] <- patid 
              mlOutcome <- merge(mlOutcome,predictionCV_LR[,c("patid",paste("predLogRegression_",nameExp[idxExp],sep=""))], by="patid")
            }
            
            if(includeSVM){
              names(predictionCV_SVM)[which(names(predictionCV_SVM) == p)] <- paste("predSVM_",nameExp[idxExp],sep="")
              # names(predictionCV_SVM)[which(names(predictionCV_SVM) == "patid")] <- patid 
              mlOutcome <- merge(mlOutcome, predictionCV_SVM[,c("patid",paste("predSVM_",nameExp[idxExp],sep=""))], by="patid")
            }
          }
          # end cross-validation type
          # ROC for CV predictions 
          # filenamePrediction <- paste( "predictions_",filenameSuffix,"_",cv,".csv",sep="" )
          # write.csv(file=paste(outputDir,filenamePrediction,sep=""), predictionCV, row.names=FALSE )
          end.time <- Sys.time()
          time.taken <- end.time - start.time
          print(paste("time taken doing CV: " , time.taken, sep=""))
          names(mlOutcome)[which(names(mlOutcome) == "patid")] <- patid
         
          write.csv(file = paste("predictions_",filenameSuffix,"_",cv,".csv",sep="" ), x= merge(dataML,mlOutcome, by=patid), row.names=FALSE)
        
          
          ### Figures and predictions experiment
          
          dataML_ROC <- dataML[ , c("Site", "patid" ,"IdwithVes", "s_pf_tpd(gVes).PRONE_PLUS", "s_pf_tpd(gVes).View1", "isch_pf_tpd(gVes).V1orV2assume", "Diagnosis_correct order",  "Territory") ]
          names(dataML_ROC) <- c("site", "patid", "id","c_sTPD","sTPD","iTPD","Diagnosis", "Territory" )
          
          dataML_ROC$DiagnosisOrg  <- as.character(dataML_ROC$Diagnosis)
          
          #### FIGURES
          dataML_ROC$DiagnosisOrg <- as.integer(dataML_ROC$DiagnosisOrg)
          #
          names(predictionCV)[2] <- "ML" # -- ojo, HARD CODED
          names(predictionCV)[which(names(predictionCV) == "patid")] = "id"
          
          predictionCV <- merge(predictionCV, dataML_ROC, by="id")
          
          
          #
          idx<-which(is.na(predictionCV$Diagnosis))
          print(predictionCV$patid[idx])
          count(predictionCV[idx,"outcome"])
          #
          # Per vessel
          #
          dataROC<- predictionCV[which(!is.na(predictionCV$Diagnosis)),]
          FileName <- paste(outputDir[n_outcome],"ROC1_",nameExp$nameExperiment[idxExp],sep="")
          dataROC$outcome <- ifelse(dataROC$outcome == 1, 1, 0)
          stats <- MultiPredROC(dataIn = dataROC,
                                outcome = "outcome",
                                pred=c("ML", "c_sTPD","sTPD","iTPD"),
                                filename=FileName,
                                labels=c("Machine learning","Combined sTPD","sTPD", "iTPD"),
                                title=paste(leadingTitleROC[n_outcome], " (n = ",as.character(length(which(dataROC$outcome==1))),"/", as.character(dim(dataROC)[1]),")",sep=""),
                                colors=c("red","dodgerblue", "dodgerblue4","midnightblue"),
                                decimals=3
          )
          #
          write.csv(x=stats$AUC, file=paste(outputDir[n_outcome],"AUC11_",nameExp$nameExperiment[idxExp],".csv",sep=""), row.names = F)
          write.csv(x=stats$Delong, file=paste(outputDir[n_outcome],"DeLong1_",nameExp$nameExperiment[idxExp],".csv",sep=""))
          #
          #Original data with predictions
          names(predictionCV)[1] <- 'IdwithVes'
          
          
          dataMLOr <- read.csv( file=dataFileName[n_outcome], stringsAsFactors = FALSE, check.names = FALSE)# entire clean datase, final, used in WEKA
          #
          # write.csv(x=merge(dataMLOr[,c("masterID",toUse,"Revascularization")], predictionCV, by="masterID"), file=paste(outputDir,"dataML.csv",sep=""), row.names = F, na="")
          write.csv(x=merge(dataMLOr[,c("Site", "patid",toUse,outcomeNameVector)], predictionCV, by = c( "IdwithVes")),
                    file=paste(outputDir[n_outcome],"dataML.csv",sep=""), row.names = F, na="")
          
          
          predictionCV_rshp <- reshape(predictionCV,idvar = c("site", "patid"),  timevar = "Territory", direction="wide" )
          
          
          ### LAD
          dataROC<- predictionCV_rshp[which(!is.na(predictionCV_rshp$Diagnosis.LAD)),]
          FileName <- paste(outputDir[n_outcome],"ROC1_LAD_",nameExp$nameExperiment[idxExp],sep="")
          dataROC$outcome <- ifelse(dataROC$outcome.LAD == 1, 1, 0)
          stats <- MultiPredROC(dataIn = dataROC,
                                outcome = "outcome",
                                pred=c("ML.LAD", "c_sTPD.LAD","sTPD.LAD","iTPD.LAD"),
                                filename=FileName,
                                labels=c("Machine learning","Combined sTPD","sTPD", "iTPD"),
                                title=paste(leadingTitleROC[n_outcome], " LAD (n = ",as.character(length(which(dataROC$outcome==1))),"/", as.character(dim(dataROC)[1]),")",sep=""),
                                colors=c("red","dodgerblue", "dodgerblue4","midnightblue"),
                                decimals=3
          )
          #
          write.csv(x=stats$AUC, file=paste(outputDir[n_outcome],"AUC11_LAD_",nameExp$nameExperiment[idxExp],".csv",sep=""), row.names = F)
          write.csv(x=stats$Delong, file=paste(outputDir[n_outcome],"DeLong1_LAD_",nameExp$nameExperiment[idxExp],".csv",sep=""))
          ### LCx
          dataROC<- predictionCV_rshp[which(!is.na(predictionCV_rshp$Diagnosis.LCx)),]
          FileName <- paste(outputDir[n_outcome],"ROC1_LCx_",nameExp$nameExperiment[idxExp],sep="")
          dataROC$outcome <- ifelse(dataROC$outcome.LCx == 1, 1, 0)
          stats <- MultiPredROC(dataIn = dataROC,
                                outcome = "outcome",
                                pred=c("ML.LCx", "c_sTPD.LCx","sTPD.LCx","iTPD.LCx"),
                                filename=FileName,
                                labels=c("Machine learning","Combined sTPD","sTPD", "iTPD"),
                                title=paste(leadingTitleROC[n_outcome], " LCx (n = ",as.character(length(which(dataROC$outcome==1))),"/", as.character(dim(dataROC)[1]),")",sep=""),
                                colors=c("red","dodgerblue", "dodgerblue4","midnightblue"),
                                decimals=3
          )
          #
          write.csv(x=stats$AUC, file=paste(outputDir[n_outcome],"AUC11_LCx_",nameExp$nameExperiment[idxExp],".csv",sep=""), row.names = F)
          write.csv(x=stats$Delong, file=paste(outputDir[n_outcome],"DeLong1_LCx_",nameExp$nameExperiment[idxExp],".csv",sep=""))
          ### RCA
          dataROC<- predictionCV_rshp[which(!is.na(predictionCV_rshp$Diagnosis.LCx)),]
          FileName <- paste(outputDir[n_outcome],"ROC1_RCA_",nameExp$nameExperiment[idxExp],sep="")
          dataROC$outcome <- ifelse(dataROC$outcome.RCA == 1, 1, 0)
          stats <- MultiPredROC(dataIn = dataROC,
                                outcome = "outcome",
                                pred=c("ML.RCA", "c_sTPD.RCA","sTPD.RCA","iTPD.RCA"),
                                filename=FileName,
                                labels=c("Machine learning","Combined sTPD","sTPD", "iTPD"),
                                title=paste(leadingTitleROC[n_outcome], " RCA (n = ",as.character(length(which(dataROC$outcome==1))),"/", as.character(dim(dataROC)[1]),")",sep=""),
                                colors=c("red","dodgerblue", "dodgerblue4","midnightblue"),
                                decimals=3
          )
          #
          write.csv(x=stats$AUC, file=paste(outputDir[n_outcome],"AUC11_RCA_",nameExp$nameExperiment[idxExp],".csv",sep=""), row.names = F)
          write.csv(x=stats$Delong, file=paste(outputDir[n_outcome],"DeLong1_RCA_",nameExp$nameExperiment[idxExp],".csv",sep=""))  
          
          
          ### Per patient max
          predictionCV_rshp <- predictionCV_rshp[which(!is.na(predictionCV_rshp$Diagnosis.LCx)),]
          predictionCV_rshp$ML_Patient <- apply(predictionCV_rshp[,c("ML.LAD", "ML.LCx", "ML.RCA")], 1, max)
          predictionCV_rshp$outcome.patient <- apply(predictionCV_rshp[,c("outcome.LAD","outcome.LCx", "outcome.RCA")], 1, max)
          predictionCV_rshp$sTPD <- apply(predictionCV_rshp[,c("sTPD.LAD","sTPD.LCx", "sTPD.RCA")], 1, sum)
          predictionCV_rshp$c_sTPD <- apply(predictionCV_rshp[,c("c_sTPD.LAD","c_sTPD.LCx", "c_sTPD.RCA")], 1, sum)
          predictionCV_rshp$iTPD <- apply(predictionCV_rshp[,c("iTPD.LAD","iTPD.LCx", "iTPD.RCA")], 1, sum)
          predictionCV_rshp$Diagnosis <- apply(predictionCV_rshp[,c("Diagnosis.LAD","Diagnosis.LCx", "Diagnosis.RCA")], 1, max)
          
          dataROC <- predictionCV_rshp
          FileName <- paste(outputDir[n_outcome],"ROC1_PerPatient_",nameExp$nameExperiment[idxExp],sep="")
          dataROC$outcome <- ifelse(dataROC$outcome.patient == 1, 1, 0)
          stats <- MultiPredROC(dataIn = dataROC,
                                outcome = "outcome",
                                pred=c("ML_Patient", "c_sTPD","sTPD","iTPD", "Diagnosis"),
                                filename=FileName,
                                labels=c("Machine learning","Combined sTPD","sTPD", "iTPD", "Diagnosis"),
                                title=paste(leadingTitleROC[n_outcome], " (n = ",as.character(length(which(dataROC$outcome==1))),"/", as.character(dim(dataROC)[1]),")",sep=""),
                                colors=c("red","dodgerblue", "dodgerblue4","midnightblue", "maroon"),
                                decimals=3
          )
          #
          write.csv( x=stats$AUC, file = paste(outputDir[n_outcome],"AUC11_PerPatient_",nameExp$nameExperiment[idxExp],".csv",sep=""), row.names = F )
          write.csv( x=stats$Delong, file = paste(outputDir[n_outcome],"DeLong1_PerPatient_",nameExp$nameExperiment[idxExp],".csv",sep="") )  
          
          write.csv( file = paste(outputDir[n_outcome],"predictions_",nameExp$nameExperiment[idxExp],"_PerPatient.csv",sep=""), x = predictionCV_rshp, row.names=FALSE  )
          
          }# end for experiments
        
    } # end for follow-up
  } # end seeds
  
 
  
  
  
  
  n_outcome <- n_outcome+1
  
  
  

} # end outcomeName loop







