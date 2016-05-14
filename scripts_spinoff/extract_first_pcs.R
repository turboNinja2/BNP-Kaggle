
set.seed(1)
source('dataframe.R')
source('./common_tools.R')
require(readr)

train_set <- read_csv(file = '..\\input\\train.csv')
test_set <- read_csv(file = '..\\input\\test.csv')

str(train_set)

numeric_columns <- dataframe.numeric_columns(train_set)

train_set <- dataframe.keep(df = train_set, numeric_columns)
test_set <- dataframe.keep(df = test_set, numeric_columns)

train_set <- dataframe.drop(df = train_set, c("target","ID"))
test_set <- dataframe.drop(df = test_set, c("ID"))

print(dim(train_set))
print(dim(test_set))

all_data <- rbind.data.frame(train_set,test_set)

print(dim(all_data))

require(HotDeckImputation)

any(is.na(all_data))

all_data <- as.matrix(all_data)
all_data <- impute.mean(DATA = all_data)

any(is.na(all_data))

train <- all_data[1:nrow(train_set),]
test <- all_data[-(1:nrow(train_set)),]

print(dim(train_set))
print(dim(test_set))

pca_model <- prcomp(x = train,scale. = T, center = T)

pca_train <- predict(pca_model,newdata = train)
pca_test <- predict(pca_model,newdata = test)

print(dim(pca_train))
print(dim(pca_test))

write_csv(x = data.frame(pca_train[,1]),path = '..\\gen_data\\pca_axis1.train.csv',col_names = F)
write_csv(x = data.frame(pca_test[,1]),path = '..\\gen_data\\pca_axis1.test.csv',col_names = F)
