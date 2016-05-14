require(readr)

train = read_csv('../input/train.csv')

dataframe.drop <- function(df,drops) { df[,!(names(df) %in% drops)] }

# Extra features

train[is.na(train)]   <- -1

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in names(train)) {
  if (class(train[[f]])=="character") {
    print(f)
    levels <- unique(train[[f]])
    if(length(levels)==2)
    {
      print(paste0(f,'-> binary feature'))
    }
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
  }
}

train <- dataframe.drop(train,c("ID"))
target <- train[,"target"]
train <- dataframe.drop(train,c("target"))

require(randomForest)

model <- randomForest(x = train,y = target, ntree = 20, do.trace = T)

ordered_vars<-rownames(model$importance)[order(-model$importance)]

write_csv(x = data.frame(ordered_vars),path = '.\\importances.txt')
