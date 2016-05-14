dataframe.drop <- function(df,drops) { df[,!(names(df) %in% drops)] }

require(readr)

train_set <- read_csv('../input/train.csv')

print(dim(train_set))

str(train_set)

train_set[is.na(train_set)] <- -1

train_set <- train_set[sample(1:nrow(train_set),size = 10000),]

labels <- train_set$target

train_set <- dataframe.drop(df = train_set, c("target"))

print(dim(train_set))

train_set <- model.matrix(~ .-1, data = train_set)

print(dim(train_set))

not_almost_constant_columns <- which(apply(train_set, 2, var, na.rm=TRUE) > 0.00015)
train_set <- train_set[,not_almost_constant_columns]

print(dim(train_set))

require(glmnet)

glmnet_model <- cv.glmnet(x = train_set,y = as.matrix(labels),type.measure = 'deviance')

plot(glmnet_model)

