set.seed(1)
source('./common_tools.R')

stage0 <- assemble_features('../gen_data/')
tmp <- import_labels()

require(corrplot)


pdf('../tmp/cor_stage0.pdf')
corrplot(cor(stage0$train))
dev.off()
