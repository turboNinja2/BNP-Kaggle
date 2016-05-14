require(readr)

train_set <- read_csv('../input/train.csv')

level_count <- function(x) {length(unique(x))}

any_na <- function(x) {any(is.na(x))}

coltypes <- sapply(train_set, class)
colnames <- names(train_set)
collevels <- sapply(train_set, level_count)
colnas <- sapply(train_set, any_na)
colmin <- sapply(train_set, min)
colmax <- sapply(train_set, max)

result <- data.frame(Name = colnames, Type = coltypes, Level = collevels, NAs = colnas, Min = colmin, Max = colmax)

result <- result[order(-result$Level),]
result <- result[order(result$Type),]

write_csv(x = result,path = ".//types.csv")
