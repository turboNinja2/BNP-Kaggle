is.constant <- function(x) {return(length(unique(x))==1)}

is.binary <- function(x) {return(length(unique(x))==2)}

level_count <- function(x) {length(unique(x))}

any_na <- function(x) {any(is.na(x))}

dataframe.drop <- function(df,drops) { df[,!(names(df) %in% drops)] }

dataframe.keep <- function(df,keeps) { df[,(names(df) %in% keeps)] }

dataframe.constant_columns <- function(df) { names(which(apply(df, 2, is.constant))) }

daraframe.low_variance <- function(df, threshold = 0.001) { names(which(apply(df, 2, var)<threshold)) }

dataframe.binary_columns <- function(df) { names(which(apply(df, 2, is.binary))) }

dataframe.summarize_in_file <- function(df, filename)
{
    coltypes <- sapply(df, class)
    colnames <- names(df)
    collevels <- sapply(df, level_count)
    colnas <- sapply(df, any_na)
    colmin <- sapply(df, min)
    colmax <- sapply(df, max)

    result <- data.frame(Name = colnames, Type = coltypes, Level = collevels, NAs = colnas, Min = colmin, Max = colmax)

    result <- result[order(-result$Level),]
    result <- result[order(result$Type),]

    write_csv(x = result,path = filename)
}

dataframe.duplicated_columns <- function(df) {
  features_pair <- combn(names(df), 2, simplify = F)
  toRemove <- c()
  for(pair in features_pair) {
    f1 <- pair[1]
    f2 <- pair[2]
    if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
      if (all(df[[f1]] == df[[f2]])) {
        toRemove <- c(toRemove, f2)
      }
    }
  }
  return(toRemove)
}

dataframe.highly_correlated_columns <- function(df,threshold=0.995) {
  features_pair <- combn(names(df), 2, simplify = F)
  toRemove <- c()
  for(pair in features_pair) {
    f1 <- pair[1]
    f2 <- pair[2]
    if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
      if (cor(df[[f1]],df[[f2]])>threshold) {
        toRemove <- c(toRemove, f2)
      }
    }
  }
  return(toRemove)
}

dataframe.numeric_columns <- function(df)
{
  nums <- sapply(train_set, is.numeric)
  return(names(nums[nums]))
  
  
}