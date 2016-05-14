assemble_features <- function(features_path)
{
  require(readr)
  
  train_features <- list.files(features_path,pattern = "*.train.csv")
  test_features <- list.files(features_path,pattern = "*.test.csv")

  train_set <- read_csv(paste0(features_path,train_features[1]),col_names = train_features[1])
  test_set <- read_csv(paste0(features_path,test_features[1]),col_names = test_features[1])
  
  for(i in 2:length(train_features))
  {
    raw_train <- read_csv(paste0(features_path,train_features[i]),col_names = train_features[i])
    train_set <- cbind.data.frame(train_set,raw_train)
  }
  
  for(i in 2:length(test_features))
  {
    raw_test <- read_csv(paste0(features_path,test_features[i]),col_names = test_features[i])
    test_set <- cbind.data.frame(test_set,raw_test)
  }
  
  return(list(train=train_set,test=test_set))
}

import_labels <- function(folder_path)
{
  require(readr)
  
  relevance <- read_csv('../gen_data/relevance.csv',col_names = F)
  test_ids <- read_csv('../gen_data/test_ids.csv',col_names = F)
  
  return(list(test_ids=test_ids,relevance=relevance))
}

