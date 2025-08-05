library(tidyverse)
library(lme4)
library(lmerTest)

dat = read.csv("./icomp_ratio.tsv", sep = "\t")
dat$fix = factor(dat$fix, levels = c('False', 'True'), order = T)

glmm_md0 = glmer(data = dat, fix ~ arte + vein + gm + wm + csf + (1|sub),
                 family = "binomial")
summary(glmm_md0)

lmm_md_vein = lmer(dat = dat, vein ~ fix + (1|sub))
summary(lmm_md_vein)
boot_vein = bootMer(
  lmm_md_vein, FUN = fixef, nsim = 10000, type = "parametric",
  parallel = "multicore", ncpus = 6)
confint(boot_vein, method = "perc", level = 1 - (0.05 / 5))

lmm_md_arte = lmer(dat = dat, arte ~ fix + (1|sub))
summary(lmm_md_arte)
boot_arte = bootMer(
  lmm_md_arte, FUN = fixef, nsim = 10000, type = "parametric",
  parallel = "multicore", ncpus = 6)
confint(boot_arte, method = "perc", level = 1 - (0.05 / 5))

lmm_md_gm = lmer(dat = dat, gm ~ fix + (1|sub))
summary(lmm_md_gm)
boot_gm = bootMer(
  lmm_md_gm, FUN = fixef, nsim = 10000, type = "parametric",
  parallel = "multicore", ncpus = 6)
confint(boot_gm, method = "perc", level = 1 - (0.05 / 5))

lmm_md_wm = lmer(dat = dat, wm ~ fix + (1|sub))
summary(lmm_md_wm)
boot_wm = bootMer(
  lmm_md_wm, FUN = fixef, nsim = 10000, type = "parametric",
  parallel = "multicore", ncpus = 6)
confint(boot_wm, method = "perc", level = 1 - (0.05 / 5))

lmm_md_csf = lmer(dat = dat, csf ~ fix + (1|sub))
summary(lmm_md_csf)
boot_csf = bootMer(
  lmm_md_csf, FUN = fixef, nsim = 10000, type = "parametric",
  parallel = "multicore", ncpus = 6)
confint(boot_csf, method = "perc", level = 1 - (0.05 / 5))
