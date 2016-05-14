require(readr)
set.seed(1)

dataframe.drop <- function(df,drops) { df[,!(names(df) %in% drops)] }

cat("reading the train data\n")
train <- read_csv("../input/train.csv")
train[is.na(train)]   <- -1

train <- train[sample(nrow(train),size = 40000),]
print(dim(train))

print(names(train))

print(dim(train))

# removing the date column
train <- dataframe.drop(train,c("ID","v22"))

print(dim(train))

for (f in names(train)) {
  if (class(train[[f]])=="character") {
    levels <- unique(train[[f]])
    if(length(levels)>1){
    train[[f]] <- factor(train[[f]], levels=levels)
    }
    else{
      train[[f]] <- NULL
    }
  }
}


var_names <- names(train)

res <- data.frame(Var1 = "0", Var2 = "0", Res = 0)

for(name1 in var_names)
{
  if(name1!="target"&& is.factor((train[[name1]])))
  {
	for(name2 in var_names)
	{
	  if(!is.factor((train[[name2]]))){next}
	  
    if(name1==name2) {break}
    if(name2!="target")
	  {
	    str_formula_interaction <- paste0("target~",name1,"*",name2)
	    str_formula_no_interaction <- paste0("target~",name1,"+",name2)
	    
      model_linear_interaction <- lm(formula=str_formula_interaction,data = train)
	    model_linear_no_interaction <- lm(formula=str_formula_no_interaction,data = train)
	    
      r2_interaction <- summary(model_linear_interaction)$adj.r.squared
	    r2_no_interaction <- summary(model_linear_no_interaction)$adj.r.squared
	    
      score <- r2_interaction-r2_no_interaction
      
      print(paste0(name1,' ',name2,' -> ',score))
      
      res = rbind.data.frame(data.frame(Var1 = name1, Var2 = name2, Res = score),res)
	  }
	}
	res <- res[order(-res$Res),]
	write_csv(x = res, path = '.\\interactions.csv')
	
  }
  
  
}
