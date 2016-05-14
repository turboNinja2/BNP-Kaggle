require(readr)

train = read_csv('../input/train.csv')

dataframe.drop <- function(df,drops) { df[,!(names(df) %in% drops)] }

# Extra features

train[is.na(train)]   <- -1

train <- dataframe.drop(train,c("ID"))

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

train$Sum <- apply(train, 1, function(z) sum(z))


produceAnalysis <- function(train,outFileName)
{
  pdf(paste0('./',outFileName))
  
  for(name in names(train))
  {
    if(typeof(train[,name])!="character")
    {
      p0 <- hist(train[,name],plot = F)
      p1 <- hist(train[train[,"target"]==0,name],plot = F, breaks = p0$breaks)
      p2 <- hist(train[train[,"target"]==1,name],plot = F, breaks = p0$breaks)
    
      par(mfrow=c(1,2))
      
      plot( p1, col=rgb(0,0,1,1/4), main = name,xlab = "target=0")  # first histogram
      plot( p2, col=rgb(1,0,0,1/4), main = name, xlab = "target=1")  # second
      
      if(length(table(train[,name]))<100)
      {
        par(mfrow=c(1,1))
        avg = tapply(train[,"target"], train[,name], mean)
        indexes = tapply(train[,name], train[,name], mean)
        sdev = tapply(train[,"target"], train[,name], sd) / sqrt(tapply(train[,"target"], train[,name], length))
        plot(indexes, avg, main=name,ylab = "Average label value", type = "p",col="red")
        
        #arrows(indexes, avg-sdev, indexes, avg+sdev, length=0.05, angle=90, code=3)
      }
    }
  }
  dev.off()
}

produceAnalysis(train,'mainAnalysis.pdf')
