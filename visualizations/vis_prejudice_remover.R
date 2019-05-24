# visualizations for prejudice remover output
library(readr)
library(ggplot2)
library(plyr)

flag = TRUE
base_df <- read_csv('datasets/prejremover-baseline.csv')
pr_df <- read_csv('datasets/prejremover-reg.csv')
cnames <- c('Black Male','Black Female','White Male','White Female','Latino Male','Latino Female','Other Male','Other Female')

if(flag==TRUE) { # weird bugs with 0 as class. wrap to make sure we only change it once
  base_df$class = base_df$class+1
  pr_df$class = base_df$class+1
  flag = FALSE
}

#base_cnums <- table(base_df$class)
#base_cw <- 1 / base_cnums # each class weight sums to 1
#base_df$weight <- base_cw[base_df$class]
base_df$class_name <- cnames[base_df$class]
base_mu <- ddply(base_df, "class_name", summarize, grp.mean=mean(score))

ggplot(base_df, aes(x=score, y=stat(width*density), fill=score, group=class_name)) + theme_bw() +
  stat_bin(bins=20) + facet_wrap(~class_name,nrow=2) + 
  scale_y_continuous(labels=scales::percent_format(accuracy=2)) +
  geom_vline(data=base_mu, aes(xintercept=grp.mean), linetype="dashed") +
  theme(plot.title = element_text(hjust = 0.5)) + ggtitle("Class Scores, Baseline")
ggsave('visualizations/baseline_class_scores.pdf')
ggsave('visualizations/baseline_class_scores.png')

pr_df$class_name <- cnames[base_df$class]
pr_mu <- ddply(pr_df, "class_name", summarize, grp.mean=mean(score))

ggplot(pr_df, aes(x=score, y=stat(width*density), group=class_name)) + theme_bw() +
  stat_bin(bins=20) + facet_wrap(~class_name, nrow=2) + 
  scale_y_continuous(labels=scales::percent_format(accuracy=2)) +
  geom_vline(data=pr_mu, aes(xintercept=grp.mean), linetype="dashed") +ylab('percent') +
  theme(plot.title = element_text(hjust = 0.5)) + ggtitle("Class Scores, Prejudice Remover")
ggsave('visualizations/pr_class_scores.pdf')
ggsave('visualizations/pr_class_scores.png')

# part 2, show training run results
library(reshape2)
avg_df <- read_csv('datasets/prejremover-average-metrics.csv')
bm_df <- read_csv('datasets/prejremover-blackmale-metrics.csv')

avg_df <- melt(avg_df, id.vars=c('eta'), variable.name = "Metric", value.name = "Score")
avgf_df <- avg_df[avg_df$eta < 250,] # filter out eta<250
avgf_df[avgf_df$Metric  != 'Balanced Accuracy',]$Score = -avgf_df[avgf_df$Metric != 'Balanced Accuracy',]$Score

ggplot(avgf_df, aes(x=eta, y=Score, col=Metric, group=Metric)) + geom_line(linetype="dashed") + theme_bw() +
  scale_y_continuous(labels=scales::percent_format(accuracy=2)) +
  ylab("Metric Value, Weighted Avg (over classes)") + xlab("eta (Prejudice Remover)")
ggsave('visualizations/pr_wavg_metrics.pdf')
ggsave('visualizations/pr_wavg_metrics.png')
  
bm_df <- melt(bm_df, id.vars=c('eta'), variable.name = 'Metric', value.name = "Score")
bmf_df <- bm_df[bm_df$eta < 250,]

ggplot(bm_df, aes(x=eta, y=Score, col=Metric, group=Metric)) + geom_line(linetype="dashed") + theme_bw() +
  scale_y_continuous(labels=scales::percent_format(accuracy=2)) +
  ylab("Metric Value, Black Males") + xlab("eta (Prejudice Remover)")
ggsave('visualizations/pr_bm_metrics.pdf')
ggsave('visualizations/pr_bm_metrics.png')
